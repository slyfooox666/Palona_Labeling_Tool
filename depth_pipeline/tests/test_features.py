from __future__ import annotations

from pathlib import Path

import numpy as np

from palona_depth.features import (
    align_relative_depth_maps,
    build_frame_work,
    enrich_temporal_features,
    local_target_depth,
    robust_depth_bounds,
)
from palona_depth.models import ControlFrame, ControlTrack, ExtractedFrame, FrameWork


def track(track_id: str, label: str, x1: int, x2: int) -> ControlTrack:
    return ControlTrack(
        track_id=track_id,
        label=label,
        confidence=0.9,
        contours_xy=(((x1, 5), (x2, 5), (x2, 25), (x1, 25)),),
    )


def test_instance_and_pair_features_are_canonical_relative_ranks(tmp_path: Path) -> None:
    depth = np.tile(np.linspace(1, 9, 64, dtype=np.float32), (32, 1))
    low, high = robust_depth_bounds([depth])
    control = ControlFrame(
        frame_index=0,
        timestamp_seconds=0.0,
        tracks=(track("p0:1", "person", 4, 16), track("p1:0", "table", 30, 55)),
    )
    extracted = ExtractedFrame(control, tmp_path / "frame.png", 0, 0.0, 0.0)
    work = build_frame_work(
        extracted,
        depth,
        None,
        video_width=64,
        video_height=32,
        normalization_low=low,
        normalization_high=high,
        alignment_tolerance_seconds=0.1,
        person_labels={"person"},
        target_labels={"table"},
    )
    assert len(work.output["instances"]) == 2
    pair = work.output["pairs"][0]
    assert pair["source_id"] == "p0:1"
    assert pair["target_id"] == "p1:0"
    assert 0 <= pair["source_depth_rank"] < pair["target_local_depth_rank"] <= 1
    assert pair["depth_order"] == "source_nearer"
    assert pair["mask_gap_2d_norm"] > 0


def test_instance_quality_uses_da3_confidence_percentile(tmp_path: Path) -> None:
    depth = np.tile(np.linspace(1, 9, 64, dtype=np.float32), (32, 1))
    confidence = np.tile(np.linspace(0, 10, 64, dtype=np.float32), (32, 1))
    low, high = robust_depth_bounds([depth])
    control = ControlFrame(
        frame_index=0,
        timestamp_seconds=0.0,
        tracks=(track("low", "person", 2, 12), track("high", "person", 50, 62)),
    )
    extracted = ExtractedFrame(control, tmp_path / "frame.png", 0, 0.0, 0.0)
    work = build_frame_work(
        extracted,
        depth,
        confidence,
        video_width=64,
        video_height=32,
        normalization_low=low,
        normalization_high=high,
        alignment_tolerance_seconds=0.1,
        person_labels={"person"},
        target_labels={"table"},
    )
    by_id = {item["track_id"]: item for item in work.output["instances"]}
    assert 0 <= by_id["low"]["depth_confidence_percentile"] < 0.5
    assert 0.5 < by_id["high"]["depth_confidence_percentile"] <= 1
    assert by_id["low"]["feature_quality"] < by_id["high"]["feature_quality"]


def test_missing_or_nonfinite_confidence_explicitly_reduces_quality(tmp_path: Path) -> None:
    depth = np.tile(np.linspace(1, 9, 64, dtype=np.float32), (32, 1))
    low, high = robust_depth_bounds([depth])
    missing_confidence_track = ControlTrack(
        track_id="person",
        label="person",
        confidence=None,
        contours_xy=(((4, 5), (20, 5), (20, 25), (4, 25)),),
    )
    control = ControlFrame(0, 0.0, (missing_confidence_track,))
    extracted = ExtractedFrame(control, tmp_path / "frame.png", 0, 0.0, 0.0)

    missing = build_frame_work(
        extracted,
        depth,
        None,
        video_width=64,
        video_height=32,
        normalization_low=low,
        normalization_high=high,
        alignment_tolerance_seconds=0.1,
        person_labels={"person"},
        target_labels={"table"},
    ).output["instances"][0]
    nonfinite = build_frame_work(
        extracted,
        depth,
        np.full_like(depth, np.nan),
        video_width=64,
        video_height=32,
        normalization_low=low,
        normalization_high=high,
        alignment_tolerance_seconds=0.1,
        person_labels={"person"},
        target_labels={"table"},
    ).output["instances"][0]

    assert 0 < nonfinite["feature_quality"] < missing["feature_quality"] < 1
    assert "depth_confidence_percentile" not in nonfinite


def test_temporal_hysteresis_proposes_review_interval(tmp_path: Path) -> None:
    frames: list[FrameWork] = []
    for index in range(16):
        timestamp = index * 0.2
        near = index < 6
        pair = {
            "source_id": "person",
            "target_id": "table",
            "depth_gap_abs": 0.02 if near else 0.8,
            "mask_gap_2d_norm": 0.01 if near else 0.5,
            "centroid_distance_2d_norm": 0.05 if near else 0.8,
            "relative_depth_velocity": 0.0,
            "proximity_score": 0.0,
            "proximity_duration_seconds": 0.0,
            "trend": "stable",
            "start_candidate_score": 0.0,
            "end_candidate_score": 0.0,
            "feature_quality": 1.0,
        }
        control = ControlFrame(index, timestamp, ())
        extracted = ExtractedFrame(control, tmp_path / f"{index}.png", index, timestamp, 0.0)
        frames.append(
            FrameWork(
                extracted=extracted,
                depth=np.ones((2, 2), dtype=np.float32),
                confidence=None,
                instances={},
                output={
                    "frame_index": index,
                    "timestamp_seconds": timestamp,
                    "instances": [],
                    "pairs": [pair],
                },
            )
        )
    candidates = enrich_temporal_features(frames)
    assert len(candidates) == 1
    assert candidates[0]["start_time"] == 0.0
    assert candidates[0]["end_time"] is not None
    assert candidates[0]["end_time"] >= 1.2
    assert frames[1].output["pairs"][0]["trend"] in {"approaching", "stable", "leaving"}


def test_temporal_hysteresis_does_not_bridge_long_missing_gap(tmp_path: Path) -> None:
    frames: list[FrameWork] = []
    for index, timestamp in enumerate((0.0, 0.2, 0.4, 0.6, 0.8, 2.0)):
        pair = {
            "source_id": "person",
            "target_id": "table",
            "depth_gap_abs": 0.02,
            "mask_gap_2d_norm": 0.01,
            "centroid_distance_2d_norm": 0.05,
            "relative_depth_velocity": 0.0,
            "proximity_score": 0.0,
            "proximity_duration_seconds": 0.0,
            "trend": "stable",
            "start_candidate_score": 0.0,
            "end_candidate_score": 0.0,
            "feature_quality": 1.0,
        }
        control = ControlFrame(index, timestamp, ())
        extracted = ExtractedFrame(control, tmp_path / f"gap-{index}.png", index, timestamp, 0.0)
        frames.append(
            FrameWork(
                extracted=extracted,
                depth=np.ones((2, 2), dtype=np.float32),
                confidence=None,
                instances={},
                output={
                    "frame_index": index,
                    "timestamp_seconds": timestamp,
                    "instances": [],
                    "pairs": [pair],
                },
            )
        )

    candidates = enrich_temporal_features(frames, max_gap_seconds=0.5)
    assert len(candidates) == 1
    assert candidates[0]["start_time"] == 0.0
    assert candidates[0]["end_time"] == 0.8
    assert frames[-1].output["pairs"][0]["proximity_duration_seconds"] == 0.0


def test_instance_velocity_does_not_bridge_long_missing_gap(tmp_path: Path) -> None:
    frames: list[FrameWork] = []
    for index, (timestamp, rank) in enumerate(((0.0, 0.1), (0.2, 0.3), (2.0, 0.9))):
        control = ControlFrame(index, timestamp, ())
        extracted = ExtractedFrame(control, tmp_path / f"velocity-{index}.png", index, timestamp, 0.0)
        frames.append(
            FrameWork(
                extracted=extracted,
                depth=np.ones((2, 2), dtype=np.float32),
                confidence=None,
                instances={},
                output={
                    "frame_index": index,
                    "timestamp_seconds": timestamp,
                    "instances": [{"track_id": "person", "depth_rank": rank, "depth_velocity": 99.0}],
                    "pairs": [],
                },
            )
        )

    enrich_temporal_features(frames, max_gap_seconds=0.5)
    assert np.isclose(frames[1].output["instances"][0]["depth_velocity"], 1.0)
    assert frames[2].output["instances"][0]["depth_velocity"] == 0.0


def test_temporal_alignment_removes_independent_affine_drift() -> None:
    base = np.tile(np.linspace(1, 9, 64, dtype=np.float32), (32, 1))
    drifted = base * 2.5 + 7.0
    aligned, metadata = align_relative_depth_maps([base, drifted])

    assert np.allclose(aligned[0], aligned[1], atol=1e-5)
    assert metadata["method"] == "per_frame_robust_affine_to_clip_anchor"
    assert len(metadata["frame_transforms"]) == 2
    assert all(0 < item["stability_quality"] <= 1 for item in metadata["frame_transforms"])


def test_temporal_alignment_uses_same_pixel_support_across_frames() -> None:
    depth = np.tile(np.linspace(1, 9, 512, dtype=np.float32), (512, 1))
    left_anchor = np.zeros_like(depth, dtype=bool)
    right_anchor = np.zeros_like(depth, dtype=bool)
    left_anchor[:, :384] = True
    right_anchor[:, 128:] = True

    aligned, metadata = align_relative_depth_maps(
        [depth.copy(), depth.copy()],
        anchor_masks=[left_anchor, right_anchor],
    )
    assert np.array_equal(aligned[0], aligned[1])
    assert metadata["anchor_region"] == "common_untracked_background_with_full_frame_fallback"
    assert not any(item["used_full_frame_fallback"] for item in metadata["frame_transforms"])


def test_temporal_alignment_full_frame_fallback_reduces_quality() -> None:
    base = np.tile(np.linspace(1, 9, 128, dtype=np.float32), (128, 1))
    empty = np.zeros_like(base, dtype=bool)
    _, metadata = align_relative_depth_maps([base, base.copy()], anchor_masks=[empty, empty])
    assert all(item["used_full_frame_fallback"] for item in metadata["frame_transforms"])
    assert all(item["stability_quality"] == 0.75 for item in metadata["frame_transforms"])


def test_temporal_alignment_penalizes_same_distribution_with_unstable_pixels() -> None:
    base = np.tile(np.linspace(1, 9, 128, dtype=np.float32), (128, 1))
    reversed_pixels = np.fliplr(base).copy()
    stable_anchor = np.ones_like(base, dtype=bool)

    _, metadata = align_relative_depth_maps(
        [base, reversed_pixels],
        anchor_masks=[stable_anchor, stable_anchor],
    )

    transforms = metadata["frame_transforms"]
    assert all(item["same_pixel_residual_iqr_units"] > 0 for item in transforms)
    assert min(item["same_pixel_correlation"] for item in transforms) <= 0
    assert max(item["stability_quality"] for item in transforms) < 0.25


def test_local_target_depth_excludes_overlapping_person_pixels() -> None:
    depth = np.full((20, 20), 9.0, dtype=np.float32)
    source = np.zeros_like(depth, dtype=bool)
    target = np.zeros_like(depth, dtype=bool)
    target[4:16, 4:16] = True
    source[4:16, 4:10] = True
    depth[source] = 1.0

    value, visible_ratio = local_target_depth(depth, target, source)
    assert value == 9.0
    assert visible_ratio == 0.5
