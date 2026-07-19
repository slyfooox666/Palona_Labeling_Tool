"""End-to-end video + Control JSON + shared DA3 sidecar builder."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Protocol

import numpy as np

from palona_depth.control import sampled_control_frames
from palona_depth.features import (
    align_relative_depth_maps,
    build_frame_work,
    contours_to_mask,
    enrich_temporal_features,
    robust_depth_bounds,
)
from palona_depth.models import DepthArtifact
from palona_depth.runtime_client import AiModelsDepthClient, load_depth_array
from palona_depth.safety import assert_artifacts_outside_git, require_private_json_suffix
from palona_depth.video import extract_aligned_frames, probe_video


Progress = Callable[[str], None]


class DepthClient(Protocol):
    def ensure_ready(self) -> dict[str, Any]: ...
    def infer_image(self, input_path: Path, output_dir: Path) -> DepthArtifact: ...
    def stop(self) -> None: ...


@dataclass(frozen=True)
class BuildOptions:
    video_path: Path
    contour_path: Path
    output_path: Path
    sample_fps: float = 5.0
    start_time: float = 0.0
    end_time: float | None = None
    max_frames: int | None = None
    device: str = "auto"
    alignment_tolerance_seconds: float | None = None
    person_labels: tuple[str, ...] = ("person",)
    target_labels: tuple[str, ...] = ("table",)
    keep_depth_artifacts: Path | None = None
    stop_runtime: bool = False


def build_depth_features(
    options: BuildOptions,
    *,
    client: DepthClient | None = None,
    progress: Progress | None = None,
) -> dict[str, Any]:
    report = progress or (lambda _: None)
    started = time.perf_counter()
    video_path = options.video_path.expanduser().resolve()
    contour_path = options.contour_path.expanduser().resolve()
    output_path = options.output_path.expanduser().resolve()
    if output_path in {video_path, contour_path}:
        raise ValueError("Depth sidecar output must not overwrite the source video or Control JSON")
    require_private_json_suffix(
        output_path,
        suffix=".depth-features.json",
        artifact_name="Depth sidecar output",
    )
    report("Probing source video")
    metadata = probe_video(video_path)
    tolerance = options.alignment_tolerance_seconds
    if tolerance is None:
        tolerance = 0.5 / metadata.fps + 1e-4
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("alignment_tolerance_seconds must be a finite positive number")
    report("Streaming and sampling Control JSON")
    controls = sampled_control_frames(
        contour_path,
        sample_fps=options.sample_fps,
        video_fps=metadata.fps,
        start_time=options.start_time,
        end_time=options.end_time,
        max_frames=options.max_frames,
    )
    report(f"Selected {len(controls)} Control frames at up to {options.sample_fps:g} FPS")

    managed_client = client or AiModelsDepthClient(device=options.device)
    temporary: tempfile.TemporaryDirectory[str] | None = None
    if options.keep_depth_artifacts is None:
        temporary = tempfile.TemporaryDirectory(prefix="palona-depth-")
        artifacts_root = Path(temporary.name)
    else:
        artifacts_root = options.keep_depth_artifacts.expanduser().resolve()
        assert_artifacts_outside_git(artifacts_root)
        artifacts_root.mkdir(parents=True, exist_ok=True)

    try:
        extracted = extract_aligned_frames(
            video_path,
            controls,
            artifacts_root / "source_frames",
            tolerance_seconds=tolerance,
        )
        health = managed_client.ensure_ready()
        report(
            f"DA3 ready on {health.get('device', 'unknown')} "
            f"({health.get('model', {}).get('model_revision', 'unknown revision')})"
        )
        inferred: list[tuple[Any, Any, Any, DepthArtifact]] = []
        for index, item in enumerate(extracted, start=1):
            artifact = managed_client.infer_image(
                item.image_path,
                artifacts_root / "runtime" / f"frame_{item.control.frame_index:08d}",
            )
            depth, confidence = load_depth_array(artifact)
            inferred.append((item, depth, confidence, artifact))
            report(f"Depth {index}/{len(extracted)} · t={item.control.timestamp_seconds:.3f}s")

        inferred_shapes = {artifact.shape for _, _, _, artifact in inferred}
        if len(inferred_shapes) != 1:
            raise ValueError(f"DA3 returned inconsistent frame shapes: {sorted(inferred_shapes)}")
        inference_configs = {
            (
                artifact.model.get("model_name"),
                artifact.model.get("model_revision"),
                artifact.model.get("device"),
                artifact.model.get("dtype"),
            )
            for _, _, _, artifact in inferred
        }
        if len(inference_configs) != 1:
            raise ValueError("DA3 changed model revision, device, or dtype within one sidecar job")
        depth_height, depth_width = next(iter(inferred_shapes))
        background_anchors = []
        for item, _depth, _confidence, _artifact in inferred:
            background = np.ones((depth_height, depth_width), dtype=bool)
            for track in item.control.tracks:
                background &= ~contours_to_mask(
                    track,
                    video_width=metadata.width,
                    video_height=metadata.height,
                    depth_width=depth_width,
                    depth_height=depth_height,
                )
            background_anchors.append(background)
        aligned_depths, temporal_alignment = align_relative_depth_maps(
            [depth for _, depth, _, _ in inferred],
            anchor_masks=background_anchors,
        )
        for record, transform in zip(inferred, temporal_alignment["frame_transforms"], strict=True):
            transform["frame_index"] = record[0].control.frame_index
            transform["timestamp_seconds"] = record[0].control.timestamp_seconds
        inferred = [
            (item, aligned_depth, confidence, artifact)
            for (item, _raw_depth, confidence, artifact), aligned_depth in zip(
                inferred,
                aligned_depths,
                strict=True,
            )
        ]
        low, high = robust_depth_bounds(depth for _, depth, _, _ in inferred)
        person_labels = {value.casefold().strip() for value in options.person_labels if value.strip()}
        target_labels = {value.casefold().strip() for value in options.target_labels if value.strip()}
        if not person_labels or not target_labels:
            raise ValueError("At least one person label and target label are required")
        frame_work = [
            build_frame_work(
                item,
                depth,
                confidence,
                video_width=metadata.width,
                video_height=metadata.height,
                normalization_low=low,
                normalization_high=high,
                alignment_tolerance_seconds=tolerance,
                person_labels=person_labels,
                target_labels=target_labels,
                temporal_alignment_quality=float(
                    temporal_alignment["frame_transforms"][index]["stability_quality"]
                ),
            )
            for index, (item, depth, confidence, _) in enumerate(inferred)
        ]
        max_temporal_gap = max(0.5, 1.5 / options.sample_fps)
        candidates = enrich_temporal_features(frame_work, max_gap_seconds=max_temporal_gap)
        first_artifact = inferred[0][3]
        first_depth_shape = list(first_artifact.shape)
        payload = {
            "schema_version": "palona.depth-features/v1",
            "created_at": datetime.now(UTC).isoformat(),
            "video": video_path.name,
            "contour": contour_path.name,
            "source": {
                "video_width": metadata.width,
                "video_height": metadata.height,
                "video_fps": metadata.fps,
                "video_duration_seconds": metadata.duration_seconds,
                "video_frame_count": metadata.frame_count,
                "video_file_size_bytes": video_path.stat().st_size,
                "contour_file_size_bytes": contour_path.stat().st_size,
            },
            "depth_metadata": {
                "model": first_artifact.model.get("model_name", "depth-anything/DA3-BASE"),
                "model_revision": first_artifact.model.get("model_revision"),
                "device": first_artifact.model.get("device", health.get("device")),
                "dtype": first_artifact.model.get("dtype", health.get("dtype")),
                "metric": False,
                "metric_units": None,
                "raw_depth_direction": "larger_is_farther",
                "depth_semantics": "depth_rank: 0=near, 1=far",
                "inference_mode": "independent_depth_image",
                "temporal_alignment": temporal_alignment,
                "normalization": {
                    "method": "clip_robust_quantile",
                    "low_quantile": 0.02,
                    "high_quantile": 0.98,
                    "aligned_low": low,
                    "aligned_high": high,
                },
                "sample_fps": options.sample_fps,
                "sample_count": len(frame_work),
                "max_alignment_error_seconds": tolerance,
                "max_cue_age_seconds": 0.5 / options.sample_fps + tolerance,
                "max_temporal_gap_seconds": max_temporal_gap,
                "process_size_hw": first_depth_shape,
                "coordinate_transform": {
                    "method": "independent_xy_scale",
                    "scale_x": first_depth_shape[1] / metadata.width,
                    "scale_y": first_depth_shape[0] / metadata.height,
                },
                "instance_statistic": "median_inside_eroded_contour_mask",
                "target_local_statistic": "median_of_nearest_10_percent_target_mask_pixels",
                "heuristic_notice": "Boundary candidates are review cues, not automatic interaction labels.",
            },
            "feature_config": {
                "person_labels": sorted(person_labels),
                "target_labels": sorted(target_labels),
                "start_threshold": 0.68,
                "end_threshold": 0.45,
                "start_hold_seconds": 0.6,
                "end_hold_seconds": 0.8,
                "max_missing_gap_seconds": max_temporal_gap,
            },
            "frames": [item.output for item in frame_work],
            "boundary_candidates": candidates,
            "processing": {
                "total_seconds": time.perf_counter() - started,
                "raw_artifacts_retained": options.keep_depth_artifacts is not None,
            },
        }
        validate_payload(payload)
        atomic_write_json(output_path, payload)
        report(f"Wrote {output_path}")
        return payload
    finally:
        if options.stop_runtime:
            managed_client.stop()
        if temporary is not None:
            temporary.cleanup()


def validate_payload(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != "palona.depth-features/v1":
        raise ValueError("Invalid depth sidecar schema version")
    metadata = payload.get("depth_metadata")
    if not isinstance(metadata, dict) or metadata.get("metric") is not False:
        raise ValueError("Depth sidecar must explicitly declare metric=false")
    if metadata.get("depth_semantics") != "depth_rank: 0=near, 1=far":
        raise ValueError("Depth sidecar must use canonical near/far rank semantics")
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Depth sidecar must contain at least one frame")
    previous_time = -1.0
    for frame in frames:
        timestamp = float(frame["timestamp_seconds"])
        if timestamp < previous_time:
            raise ValueError("Depth sidecar frames must be time ordered")
        previous_time = timestamp
        for instance in frame.get("instances", []):
            rank = float(instance["depth_rank"])
            quality = float(instance["feature_quality"])
            if not (math_is_finite(rank) and 0 <= rank <= 1):
                raise ValueError("Instance depth_rank must be finite and within [0,1]")
            if not (math_is_finite(quality) and 0 <= quality <= 1):
                raise ValueError("Instance feature_quality must be within [0,1]")
        for pair in frame.get("pairs", []):
            for key in ("depth_gap_abs", "proximity_score", "feature_quality"):
                value = float(pair[key])
                if not math_is_finite(value):
                    raise ValueError(f"Pair {key} must be finite")


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    temporary.replace(path)


def math_is_finite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}
