"""Tracklet evidence collection and frame selection."""

from __future__ import annotations

from dataclasses import dataclass, field

from vision_pipeline.core.schemas import BoundingBox, CropReference


@dataclass(frozen=True)
class FrameEvidence:
    """One candidate evidence frame for a tracked object/person."""

    timestamp_seconds: float
    bbox: BoundingBox
    confidence: float
    frame_uri: str | None = None
    crop_refs: tuple[CropReference, ...] = ()
    event_tags: tuple[str, ...] = ()
    blur_score: float | None = None
    occlusion_score: float | None = None

    @property
    def quality_score(self) -> float:
        """Score used for selecting useful VLM evidence frames."""
        blur = self.blur_score if self.blur_score is not None else 100.0
        occlusion_penalty = self.occlusion_score if self.occlusion_score is not None else 0.0
        return self.confidence + (blur / 100.0) - occlusion_penalty


@dataclass
class TrackletEvidenceBuffer:
    """Keeps track observations and selects a small VLM evidence bundle."""

    track_id: str
    frames: list[FrameEvidence] = field(default_factory=list)

    def add(self, frame: FrameEvidence) -> None:
        self.frames.append(frame)
        self.frames.sort(key=lambda item: item.timestamp_seconds)

    def select_frames(
        self,
        max_frames: int,
        preferred_event_tags: tuple[str, ...] = (),
    ) -> list[FrameEvidence]:
        """Select diverse, high-signal frames for VLM classification."""
        if max_frames <= 0 or not self.frames:
            return []

        selected: list[FrameEvidence] = []
        selected_ids: set[int] = set()

        for tag in preferred_event_tags:
            candidates = [
                (index, frame)
                for index, frame in enumerate(self.frames)
                if tag in frame.event_tags and index not in selected_ids
            ]
            if candidates:
                index, frame = max(candidates, key=lambda item: item[1].quality_score)
                selected.append(frame)
                selected_ids.add(index)
                if len(selected) >= max_frames:
                    return sorted(selected, key=lambda item: item.timestamp_seconds)

        remaining_slots = max_frames - len(selected)
        if remaining_slots <= 0:
            return sorted(selected, key=lambda item: item.timestamp_seconds)

        remaining = [
            (index, frame)
            for index, frame in enumerate(self.frames)
            if index not in selected_ids
        ]
        if remaining_slots >= len(remaining):
            selected.extend(frame for _, frame in remaining)
            return sorted(selected, key=lambda item: item.timestamp_seconds)

        # Encourage temporal diversity by selecting from equally spaced buckets.
        bucket_size = max(1, len(remaining) // remaining_slots)
        for start in range(0, len(remaining), bucket_size):
            bucket = remaining[start : start + bucket_size]
            if not bucket:
                continue
            index, frame = max(bucket, key=lambda item: item[1].quality_score)
            selected.append(frame)
            selected_ids.add(index)
            if len(selected) >= max_frames:
                break

        return sorted(selected, key=lambda item: item.timestamp_seconds)
