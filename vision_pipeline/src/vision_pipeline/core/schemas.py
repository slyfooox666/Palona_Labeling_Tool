"""Shared data structures used across model adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

BBoxXYXY = tuple[float, float, float, float]


@dataclass(frozen=True)
class BoundingBox:
    """Bounding box in xyxy image coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def as_xyxy(self) -> BBoxXYXY:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class Detection:
    """Single model detection."""

    bbox: BoundingBox
    label: str
    confidence: float
    model_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Track:
    """Tracker output for a single object/person at one timestamp."""

    track_id: str
    bbox: BoundingBox
    label: str
    confidence: float
    timestamp_seconds: float
    state: Literal["active", "lost", "removed"] = "active"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CropReference:
    """Reference to a crop/contact sheet produced for VLM evidence."""

    uri: str
    crop_type: Literal["tight", "context", "contact_sheet"]
    timestamp_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class VLMDecision:
    """Structured VLM decision for a tracklet."""

    track_id: str
    prompt_id: str
    model_id: str
    result: dict[str, Any]
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineEvent:
    """Event emitted by a use-case pipeline."""

    event_type: str
    timestamp_seconds: float
    track_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)
