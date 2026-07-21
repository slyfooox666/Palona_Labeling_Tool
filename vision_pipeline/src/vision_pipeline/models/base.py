"""Shared model adapter protocols."""

from __future__ import annotations

from typing import Protocol

from vision_pipeline.core.schemas import Detection, Track, VLMDecision
from vision_pipeline.core.tracklet import FrameEvidence


class Detector(Protocol):
    def detect(self, frame: object, timestamp_seconds: float) -> list[Detection]:
        """Detect objects in a frame."""


class Tracker(Protocol):
    def update(
        self,
        detections: list[Detection],
        timestamp_seconds: float,
    ) -> list[Track]:
        """Update tracks from detections."""


class VLMClassifier(Protocol):
    def classify_tracklet(
        self,
        track_id: str,
        prompt: str,
        evidence: list[FrameEvidence],
    ) -> VLMDecision:
        """Classify a tracklet from selected frame evidence."""
