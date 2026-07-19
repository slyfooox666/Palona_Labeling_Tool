from __future__ import annotations

import json
from pathlib import Path
import sys

import av
import numpy as np
import pytest

from palona_depth.full_video import (
    FullVideoError,
    FullVideoOptions,
    build_upstream_command,
    run_full_video,
    validate_control_coverage,
)


def write_video(path: Path, *, frames: int = 10, fps: int = 10) -> None:
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            image = np.full((48, 64, 3), index * 10, dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(image, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def write_control(path: Path, *, frame_indexes: range, fps: float = 10.0) -> None:
    frames = []
    for frame_index in frame_indexes:
        frames.append(
            {
                "frame_index": frame_index,
                "timestamp_seconds": frame_index / fps,
                "tracks": [
                    {
                        "track_id": "p0:1",
                        "label": "person",
                        "confidence": 0.9,
                        "contours_xy": [[[2, 2], [20, 2], [20, 30], [2, 30]]],
                    }
                ],
            }
        )
    path.write_text(json.dumps({"video": "clip.mp4", "frames": frames}), encoding="utf-8")


def options(tmp_path: Path) -> FullVideoOptions:
    return FullVideoOptions(
        video_path=tmp_path / "clip.mp4",
        control_path=tmp_path / "clip.control.json",
        output_dir=tmp_path / "artifacts",
        vision_pipeline_root=tmp_path / "vision_pipeline",
        pipeline_python=Path("/usr/bin/python3"),
        prompts=("person", "cashier machine"),
        required_labels=("person",),
    )


def test_upstream_command_is_whole_chunked_and_unbounded(tmp_path: Path) -> None:
    temporary = tmp_path / "clip.control.json.123.tmp"
    command = build_upstream_command(options(tmp_path), temporary)

    assert command[2:5] == ["sam3", str((tmp_path / "clip.mp4").resolve()), "--model-config"]
    assert command[command.index("--video-mode") + 1] == "whole"
    assert command[command.index("--split-seconds") + 1] == "12"
    assert command[command.index("--overlap-seconds") + 1] == "2"
    assert command[command.index("--prompt") + 1 : command.index("--output-dir")] == [
        "person",
        "cashier machine",
    ]
    assert "--max-frames" not in command
    assert command[-1] == "--timeit"


def test_complete_control_coverage_is_accepted(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    control = tmp_path / "clip.control.json"
    write_video(video)
    write_control(control, frame_indexes=range(10))

    coverage = validate_control_coverage(control, video, required_labels=("PERSON",))

    assert coverage.frame_count == 10
    assert coverage.first_timestamp_seconds == 0
    assert coverage.last_timestamp_seconds == pytest.approx(0.9)
    assert coverage.contour_track_count == 10
    assert coverage.labels == ("person",)


def test_incomplete_control_is_rejected(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    control = tmp_path / "clip.control.json"
    write_video(video)
    write_control(control, frame_indexes=range(4))

    with pytest.raises(FullVideoError, match="Control ends"):
        validate_control_coverage(control, video)


def test_internal_frame_gap_is_rejected(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    control = tmp_path / "clip.control.json"
    write_video(video)
    write_control(control, frame_indexes=range(10))
    payload = json.loads(control.read_text(encoding="utf-8"))
    del payload["frames"][4:7]
    control.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FullVideoError, match="missing frames between 3 and 7"):
        validate_control_coverage(control, video)


def test_control_must_start_at_frame_zero(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    control = tmp_path / "clip.control.json"
    write_video(video)
    write_control(control, frame_indexes=range(1, 10))

    with pytest.raises(FullVideoError, match="instead of frame 0"):
        validate_control_coverage(control, video)


def test_required_label_is_checked(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    control = tmp_path / "clip.control.json"
    write_video(video)
    write_control(control, frame_indexes=range(10))

    with pytest.raises(FullVideoError, match="cashier machine"):
        validate_control_coverage(control, video, required_labels=("cashier machine",))


def test_out_of_bounds_contour_is_rejected(tmp_path: Path) -> None:
    video = tmp_path / "clip.mp4"
    control = tmp_path / "clip.control.json"
    write_video(video)
    write_control(control, frame_indexes=range(10))
    payload = json.loads(control.read_text(encoding="utf-8"))
    payload["frames"][0]["tracks"][0]["contours_xy"][0][0] = [999, 2]
    control.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FullVideoError, match="out-of-bounds"):
        validate_control_coverage(control, video)


def test_successful_run_validates_then_atomically_commits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "clip.mp4"
    write_video(video)
    root = tmp_path / "vision_pipeline"
    script = root / "scripts" / "vp_cli.py"
    script.parent.mkdir(parents=True)
    script.write_text(
        """
import json
from pathlib import Path
import sys

control = Path(sys.argv[sys.argv.index('--contour-json') + 1])
frames = []
for index in range(10):
    frames.append({
        'frame_index': index,
        'timestamp_seconds': index / 10,
        'tracks': [{
            'track_id': 'p0:1',
            'label': 'person',
            'contours_xy': [[[2, 2], [20, 2], [20, 30], [2, 30]]],
        }],
    })
control.write_text(json.dumps({'frames': frames}))
""".strip(),
        encoding="utf-8",
    )
    control = tmp_path / "private" / "clip.control.json"
    output_dir = tmp_path / "private" / "artifacts"
    run_options = FullVideoOptions(
        video_path=video,
        control_path=control,
        output_dir=output_dir,
        vision_pipeline_root=root,
        pipeline_python=Path(sys.executable),
        prompts=("person",),
        required_labels=("person",),
    )
    monkeypatch.setattr("palona_depth.full_video.preflight", lambda *_args, **_kwargs: None)

    coverage = run_full_video(run_options)

    assert coverage.frame_count == 10
    assert control.is_file()
    assert not list(control.parent.glob("*.tmp"))
    validation = json.loads((output_dir / "full-video-preprocess.validation.json").read_text())
    assert validation["last_frame_index"] == 9
    command = json.loads((output_dir / "full-video-preprocess.command.json").read_text())["argv"]
    assert "--max-frames" not in command


def test_failed_retry_removes_stale_validation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    video = tmp_path / "clip.mp4"
    write_video(video)
    root = tmp_path / "vision_pipeline"
    script = root / "scripts" / "vp_cli.py"
    script.parent.mkdir(parents=True)
    script.write_text("raise SystemExit(7)\n", encoding="utf-8")
    output_dir = tmp_path / "private" / "artifacts"
    output_dir.mkdir(parents=True)
    validation = output_dir / "full-video-preprocess.validation.json"
    validation.write_text('{"frame_count": 999}\n', encoding="utf-8")
    run_options = FullVideoOptions(
        video_path=video,
        control_path=tmp_path / "private" / "clip.control.json",
        output_dir=output_dir,
        vision_pipeline_root=root,
        pipeline_python=Path(sys.executable),
        prompts=("person",),
    )
    monkeypatch.setattr("palona_depth.full_video.preflight", lambda *_args, **_kwargs: None)

    with pytest.raises(FullVideoError, match="exit code 7"):
        run_full_video(run_options)

    assert not validation.exists()
