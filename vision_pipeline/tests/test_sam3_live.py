from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from vision_pipeline.core.sam_live import (
    Sam3LiveConfig,
    Sam3TrackStitcher,
    bbox_iou,
    build_frame_windows,
)
from vision_pipeline.core.schemas import BoundingBox, Track
from vision_pipeline.models.sam3.adapter import Sam3FrameResult
from vision_pipeline.utils.video import SampledFrame


def test_build_frame_windows_uses_overlap_without_double_commit() -> None:
    frames = [
        SampledFrame(path=Path(f"frame_{index}.jpg"), timestamp_seconds=float(index))
        for index in range(0, 61, 10)
    ]
    config = Sam3LiveConfig(
        strategy="long_window",
        sample_fps=0.1,
        window_seconds=30,
        stride_seconds=20,
    )

    windows = build_frame_windows(frames, config)

    assert [(window.start_seconds, window.end_seconds) for window in windows] == [
        (0.0, 30.0),
        (20.0, 50.0),
        (40.0, 60.0),
    ]
    assert windows[1].commit_after_seconds == 30.0
    assert windows[2].commit_after_seconds == 50.0


def test_stitcher_matches_local_ids_across_overlapping_windows() -> None:
    stitcher = Sam3TrackStitcher(
        Sam3LiveConfig(
            sample_fps=1,
            dwell_threshold_seconds=999,
            match_iou_threshold=0.2,
            match_distance_threshold_px=50,
        )
    )

    first = stitcher.process_window(
        window_index=0,
        frame_results=[
            frame_result(0, [track("1", 0, 10, 10, 50, 50)]),
            frame_result(10, [track("1", 10, 12, 10, 52, 50)]),
        ],
        commit_after_seconds=float("-inf"),
    )
    second = stitcher.process_window(
        window_index=1,
        frame_results=[
            frame_result(10, [track("4", 10, 12, 10, 52, 50)]),
            frame_result(20, [track("4", 20, 14, 10, 54, 50)]),
        ],
        commit_after_seconds=10,
    )

    assert first.committed_frames[0].tracks[0].track_id == "1"
    assert second.committed_frames[0].timestamp_seconds == 20
    assert second.committed_frames[0].tracks[0].track_id == "1"
    assert second.committed_frames[0].tracks[0].metadata["local_track_id"] == "4"


def test_stitcher_emits_dwell_event_once() -> None:
    stitcher = Sam3TrackStitcher(
        Sam3LiveConfig(
            sample_fps=1,
            dwell_threshold_seconds=12,
            match_iou_threshold=0.2,
            match_distance_threshold_px=50,
        )
    )

    result = stitcher.process_window(
        window_index=0,
        frame_results=[
            frame_result(0, [track("1", 0, 10, 10, 50, 50)]),
            frame_result(10, [track("1", 10, 10, 10, 50, 50)]),
            frame_result(15, [track("1", 15, 10, 10, 50, 50)]),
            frame_result(20, [track("1", 20, 10, 10, 50, 50)]),
        ],
        commit_after_seconds=float("-inf"),
    )

    assert [event.timestamp_seconds for event in result.events] == [15]
    assert result.events[0].dwell_seconds == 15


def test_stitcher_creates_new_global_track_for_far_object() -> None:
    stitcher = Sam3TrackStitcher(
        Sam3LiveConfig(
            sample_fps=1,
            dwell_threshold_seconds=999,
            match_iou_threshold=0.2,
            match_distance_threshold_px=50,
        )
    )

    stitcher.process_window(
        window_index=0,
        frame_results=[frame_result(0, [track("1", 0, 10, 10, 50, 50)])],
        commit_after_seconds=float("-inf"),
    )
    result = stitcher.process_window(
        window_index=1,
        frame_results=[frame_result(10, [track("1", 10, 300, 300, 350, 350)])],
        commit_after_seconds=0,
    )

    assert result.committed_frames[0].tracks[0].track_id == "2"


def test_bbox_iou() -> None:
    assert bbox_iou(BoundingBox(0, 0, 10, 10), BoundingBox(5, 5, 15, 15)) == 25 / 175


def frame_result(timestamp: float, tracks: list[Track]) -> Sam3FrameResult:
    return Sam3FrameResult(
        timestamp_seconds=timestamp,
        image_path=Path(f"frame_{timestamp}.jpg"),
        tracks=tracks,
    )


def track(
    track_id: str,
    timestamp: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> Track:
    return Track(
        track_id=track_id,
        bbox=BoundingBox(x1, y1, x2, y2),
        label="basket",
        confidence=0.9,
        timestamp_seconds=timestamp,
    )
