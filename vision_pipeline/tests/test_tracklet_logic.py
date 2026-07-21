from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from vision_pipeline.core.schemas import BoundingBox
from vision_pipeline.core.tracklet import FrameEvidence, TrackletEvidenceBuffer


def test_tracklet_frame_selection_prefers_event_frames() -> None:
    buffer = TrackletEvidenceBuffer(track_id="person-1")
    for index in range(10):
        tags = ("employee_interaction",) if index == 7 else ()
        buffer.add(
            FrameEvidence(
                timestamp_seconds=float(index),
                bbox=BoundingBox(0, 0, 10, 20),
                confidence=0.5 + index * 0.01,
                event_tags=tags,
            )
        )

    selected = buffer.select_frames(
        max_frames=4,
        preferred_event_tags=("employee_interaction",),
    )

    assert len(selected) == 4
    assert any("employee_interaction" in frame.event_tags for frame in selected)
