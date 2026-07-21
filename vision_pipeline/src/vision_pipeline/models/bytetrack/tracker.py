"""ByteTrack tracker adapter skeleton."""

from __future__ import annotations

from vision_pipeline.core.schemas import Detection, Track
from vision_pipeline.models.bytetrack.schemas import ByteTrackConfig


class ByteTrackTracker:
    """Adapter boundary for ByteTrack realtime object tracking."""

    def __init__(self, config: ByteTrackConfig) -> None:
        self.config = config

    def update(
        self,
        detections: list[Detection],
        timestamp_seconds: float,
    ) -> list[Track]:
        raise NotImplementedError(
            "Wire ByteTrack runtime here and return Track objects."
        )
