from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from vision_pipeline.core.schemas import BoundingBox, Detection
from vision_pipeline.models.bot_sort.schemas import BoTSORTConfig
from vision_pipeline.models.bot_sort.tracker import BoTSORTTracker


def person_detection(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    appearance: tuple[float, ...] | None = None,
) -> Detection:
    return Detection(
        bbox=BoundingBox(x1, y1, x2, y2),
        label="person",
        confidence=0.9,
        model_id="test",
        metadata={"appearance": appearance} if appearance else {},
    )


def test_bot_sort_keeps_track_id_for_overlapping_person() -> None:
    tracker = BoTSORTTracker(
        BoTSORTConfig(
            model_id="bot_sort_test",
            new_track_threshold=0.7,
            iou_match_threshold=0.2,
        )
    )

    first = tracker.update([person_detection(0, 0, 100, 200)], timestamp_seconds=0.0)
    second = tracker.update([person_detection(10, 0, 110, 200)], timestamp_seconds=1.0)

    assert len(first) == 1
    assert len(second) == 1
    assert first[0].track_id == second[0].track_id
    assert second[0].state == "active"


def test_bot_sort_marks_missing_track_lost_then_removed() -> None:
    tracker = BoTSORTTracker(
        BoTSORTConfig(
            model_id="bot_sort_test",
            new_track_threshold=0.7,
            max_lost_seconds=1.0,
        )
    )

    tracker.update([person_detection(0, 0, 100, 200)], timestamp_seconds=0.0)
    lost = tracker.update([], timestamp_seconds=0.5)
    removed = tracker.update([], timestamp_seconds=2.0)

    assert lost[0].state == "lost"
    assert removed == []


def test_bot_sort_reactivates_lost_track_with_matching_appearance() -> None:
    tracker = BoTSORTTracker(
        BoTSORTConfig(
            model_id="bot_sort_test",
            new_track_threshold=0.7,
            iou_match_threshold=0.9,
            appearance_match_threshold=0.8,
            center_match_threshold=0.1,
            max_lost_seconds=30.0,
        )
    )
    appearance = (1.0, 0.0, 0.0)

    first = tracker.update(
        [person_detection(0, 0, 100, 200, appearance=appearance)],
        timestamp_seconds=0.0,
    )
    tracker.update([], timestamp_seconds=1.0)
    reactivated = tracker.update(
        [person_detection(90, 0, 190, 200, appearance=appearance)],
        timestamp_seconds=2.0,
    )

    active_tracks = [track for track in reactivated if track.state == "active"]
    assert len(active_tracks) == 1
    assert active_tracks[0].track_id == first[0].track_id


def test_bot_sort_does_not_reactivate_lost_track_with_different_appearance() -> None:
    tracker = BoTSORTTracker(
        BoTSORTConfig(
            model_id="bot_sort_test",
            new_track_threshold=0.7,
            iou_match_threshold=0.9,
            appearance_match_threshold=0.8,
            center_match_threshold=0.1,
            max_lost_seconds=30.0,
        )
    )

    first = tracker.update(
        [person_detection(0, 0, 100, 200, appearance=(1.0, 0.0, 0.0))],
        timestamp_seconds=0.0,
    )
    tracker.update([], timestamp_seconds=1.0)
    updated = tracker.update(
        [person_detection(90, 0, 190, 200, appearance=(0.0, 1.0, 0.0))],
        timestamp_seconds=2.0,
    )

    active_tracks = [track for track in updated if track.state == "active"]
    assert len(active_tracks) == 1
    assert active_tracks[0].track_id != first[0].track_id
