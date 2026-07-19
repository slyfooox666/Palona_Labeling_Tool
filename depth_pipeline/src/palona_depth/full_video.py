"""Safe orchestration for native, chunked SAM3 whole-video preprocessing."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
from typing import Iterable

from palona_depth.control import ControlDataError, iter_control_frames
from palona_depth.safety import assert_artifacts_outside_git, require_private_json_suffix
from palona_depth.video import VideoDataError, probe_video


class FullVideoError(RuntimeError):
    """Raised when a full-video run cannot be started or safely committed."""


@dataclass(frozen=True)
class ControlCoverage:
    frame_count: int
    track_count: int
    contour_track_count: int
    first_frame_index: int
    last_frame_index: int
    first_timestamp_seconds: float
    last_timestamp_seconds: float
    video_duration_seconds: float
    video_fps: float
    labels: tuple[str, ...]


@dataclass(frozen=True)
class FullVideoOptions:
    video_path: Path
    control_path: Path
    output_dir: Path
    vision_pipeline_root: Path
    pipeline_python: Path
    prompts: tuple[str, ...]
    required_labels: tuple[str, ...] = ()
    split_seconds: float = 12.0
    overlap_seconds: float = 2.0
    contour_epsilon_px: float = 2.0
    model_config: Path | None = None
    force: bool = False


def build_upstream_command(options: FullVideoOptions, temporary_control_path: Path) -> list[str]:
    root = options.vision_pipeline_root.expanduser().resolve()
    model_config = (
        options.model_config.expanduser().resolve()
        if options.model_config is not None
        else root / "configs" / "models" / "sam3.yaml"
    )
    return [
        str(options.pipeline_python.expanduser().resolve()),
        str(root / "scripts" / "vp_cli.py"),
        "sam3",
        str(options.video_path.expanduser().resolve()),
        "--model-config",
        str(model_config),
        "--video-mode",
        "whole",
        "--prompt",
        *options.prompts,
        "--output-dir",
        str(options.output_dir.expanduser().resolve()),
        "--contour-json",
        str(temporary_control_path.expanduser().resolve()),
        "--contour-epsilon-px",
        f"{options.contour_epsilon_px:g}",
        "--split-seconds",
        f"{options.split_seconds:g}",
        "--overlap-seconds",
        f"{options.overlap_seconds:g}",
        "--timeit",
    ]


def preflight(options: FullVideoOptions, *, require_cuda: bool = True) -> None:
    video = options.video_path.expanduser().resolve()
    control = options.control_path.expanduser().resolve()
    output_dir = options.output_dir.expanduser().resolve()
    root = options.vision_pipeline_root.expanduser().resolve()
    python = options.pipeline_python.expanduser().resolve()
    model_config = (
        options.model_config.expanduser().resolve()
        if options.model_config is not None
        else root / "configs" / "models" / "sam3.yaml"
    )

    if not video.is_file():
        raise FullVideoError(f"Video does not exist: {video}")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise FullVideoError(f"Vision pipeline Python is not executable: {python}")
    if not (root / "scripts" / "vp_cli.py").is_file():
        raise FullVideoError(f"VISION_PIPELINE_ROOT is invalid: {root}")
    if not model_config.is_file():
        raise FullVideoError(f"SAM3 model config does not exist: {model_config}")
    if not options.prompts or any(not prompt.strip() for prompt in options.prompts):
        raise FullVideoError("At least one non-empty --prompt is required")
    if options.split_seconds <= 0 or not math.isfinite(options.split_seconds):
        raise FullVideoError("--split-seconds must be finite and positive")
    if (
        options.overlap_seconds < 0
        or not math.isfinite(options.overlap_seconds)
        or options.overlap_seconds >= options.split_seconds
    ):
        raise FullVideoError("--overlap-seconds must be non-negative and smaller than --split-seconds")
    if options.contour_epsilon_px < 0 or not math.isfinite(options.contour_epsilon_px):
        raise FullVideoError("--contour-epsilon-px must be finite and non-negative")
    if control == video:
        raise FullVideoError("Control output must not overwrite the source video")
    if control.exists() and not options.force:
        raise FullVideoError(f"Control output already exists (pass --force to replace it): {control}")

    require_private_json_suffix(
        control,
        suffix=".control.json",
        artifact_name="Full-video Control JSON",
    )
    assert_artifacts_outside_git(output_dir)
    for executable in ("ffmpeg", "ffprobe"):
        if shutil.which(executable) is None:
            raise FullVideoError(f"{executable} was not found on PATH")

    probe_video(video)
    help_check = subprocess.run(
        [str(python), str(root / "scripts" / "vp_cli.py"), "sam3", "--help"],
        cwd=root.parent,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    if help_check.returncode != 0:
        detail = help_check.stderr.strip() or help_check.stdout.strip()
        raise FullVideoError(f"vision_pipeline import/preflight failed: {detail}")
    if require_cuda:
        cuda_check = subprocess.run(
            [
                str(python),
                "-c",
                (
                    "import cv2, numpy, torch, yaml; "
                    "from sam3 import model_builder; "
                    "assert hasattr(model_builder, 'build_sam3_predictor') or "
                    "hasattr(model_builder, 'build_sam3_video_predictor'); "
                    "assert torch.cuda.is_available(), "
                    "'torch.cuda.is_available() is false'; "
                    "print(torch.cuda.get_device_name(0))"
                ),
            ],
            cwd=root.parent,
            text=True,
            capture_output=True,
            timeout=60,
            check=False,
        )
        if cuda_check.returncode != 0:
            detail = cuda_check.stderr.strip() or cuda_check.stdout.strip()
            raise FullVideoError(f"Native SAM3/CUDA preflight failed in {python}: {detail}")


def validate_control_coverage(
    control_path: Path,
    video_path: Path,
    *,
    required_labels: Iterable[str] = (),
) -> ControlCoverage:
    metadata = probe_video(video_path)
    labels: set[str] = set()
    frame_count = 0
    track_count = 0
    contour_track_count = 0
    first = None
    last = None
    previous_frame_index: int | None = None
    timestamp_tolerance = max(1.5 / metadata.fps, 0.02)
    for frame in iter_control_frames(control_path, video_fps=metadata.fps):
        if first is None:
            first = frame
        if previous_frame_index is not None and frame.frame_index != previous_frame_index + 1:
            raise FullVideoError(
                f"Control is missing frames between {previous_frame_index} and {frame.frame_index}"
            )
        previous_frame_index = frame.frame_index
        last = frame
        frame_count += 1
        expected_timestamp = frame.frame_index / metadata.fps
        if abs(frame.timestamp_seconds - expected_timestamp) > timestamp_tolerance:
            raise FullVideoError(
                f"Control frame {frame.frame_index} timestamp {frame.timestamp_seconds:.6f}s "
                f"does not align with video FPS {metadata.fps:.6f}"
            )
        for track in frame.tracks:
            labels.add(track.label)
            track_count += 1
            if track.contours_xy:
                contour_track_count += 1
            for contour in track.contours_xy:
                for x, y in contour:
                    if x < -1 or y < -1 or x > metadata.width + 1 or y > metadata.height + 1:
                        raise FullVideoError(
                            f"Control frame {frame.frame_index} track {track.track_id} has "
                            f"out-of-bounds contour point ({x:.3f}, {y:.3f})"
                        )
    if first is None or last is None:
        raise FullVideoError("Generated Control JSON contains no frames")
    if first.frame_index != 0:
        raise FullVideoError(f"Control starts at frame {first.frame_index} instead of frame 0")
    edge_tolerance = max(2.0 / metadata.fps, 0.1)
    if first.timestamp_seconds > edge_tolerance:
        raise FullVideoError(
            f"Control starts at {first.timestamp_seconds:.3f}s instead of the beginning of the video"
        )
    if metadata.duration_seconds > 0 and last.timestamp_seconds < metadata.duration_seconds - edge_tolerance:
        raise FullVideoError(
            f"Control ends at {last.timestamp_seconds:.3f}s but video duration is "
            f"{metadata.duration_seconds:.3f}s"
        )
    if track_count and contour_track_count == 0:
        raise FullVideoError("Control tracks contain no usable mask contours")

    normalized_labels = {label.casefold() for label in labels}
    missing = [label for label in required_labels if label.casefold() not in normalized_labels]
    if missing:
        raise FullVideoError(f"Required Control labels were not detected: {', '.join(missing)}")

    return ControlCoverage(
        frame_count=frame_count,
        track_count=track_count,
        contour_track_count=contour_track_count,
        first_frame_index=first.frame_index,
        last_frame_index=last.frame_index,
        first_timestamp_seconds=first.timestamp_seconds,
        last_timestamp_seconds=last.timestamp_seconds,
        video_duration_seconds=metadata.duration_seconds,
        video_fps=metadata.fps,
        labels=tuple(sorted(labels)),
    )


def run_full_video(options: FullVideoOptions, *, require_cuda: bool = True) -> ControlCoverage:
    preflight(options, require_cuda=require_cuda)
    control = options.control_path.expanduser().resolve()
    output_dir = options.output_dir.expanduser().resolve()
    control.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = control.with_name(f"{control.name}.{os.getpid()}.tmp")
    log_path = output_dir / "full-video-preprocess.log"
    command_path = output_dir / "full-video-preprocess.command.json"
    validation_path = output_dir / "full-video-preprocess.validation.json"
    command = build_upstream_command(options, temporary)

    # A failed retry must not leave a previous run's validation beside a new
    # failure log, where it could be mistaken for evidence about this run.
    validation_path.unlink(missing_ok=True)
    command_path.write_text(
        json.dumps({"argv": command, "cwd": str(options.vision_pipeline_root.resolve().parent)}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    command_path.chmod(0o600)
    try:
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"$ {shlex.join(command)}\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=options.vision_pipeline_root.resolve().parent,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stderr.write(line)
                log.write(line)
            return_code = process.wait()
        log_path.chmod(0o600)
        if return_code != 0:
            raise FullVideoError(f"vision_pipeline failed with exit code {return_code}; see {log_path}")
        if not temporary.is_file():
            raise FullVideoError(f"vision_pipeline did not create temporary Control JSON: {temporary}")
        coverage = validate_control_coverage(
            temporary,
            options.video_path,
            required_labels=options.required_labels,
        )
        os.replace(temporary, control)
        control.chmod(0o600)
        validation_path.write_text(json.dumps(asdict(coverage), indent=2) + "\n", encoding="utf-8")
        validation_path.chmod(0o600)
        return coverage
    finally:
        temporary.unlink(missing_ok=True)
