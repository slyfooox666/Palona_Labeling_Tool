"""Internal normalized records shared by video, contour, and feature stages."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


Point = tuple[float, float]


@dataclass(frozen=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    duration_seconds: float
    frame_count: int | None


@dataclass(frozen=True)
class ControlTrack:
    track_id: str
    label: str
    confidence: float | None
    contours_xy: tuple[tuple[Point, ...], ...]


@dataclass(frozen=True)
class ControlFrame:
    frame_index: int
    timestamp_seconds: float
    tracks: tuple[ControlTrack, ...]


@dataclass(frozen=True)
class ExtractedFrame:
    control: ControlFrame
    image_path: Path
    decoded_frame_index: int
    decoded_timestamp_seconds: float
    alignment_error_seconds: float


@dataclass(frozen=True)
class DepthArtifact:
    depth_path: Path
    confidence_path: Path | None
    shape: tuple[int, int]
    model: dict[str, Any]
    processing: dict[str, Any]


@dataclass
class InstanceWork:
    output: dict[str, Any]
    mask: np.ndarray
    raw_depth_median: float


@dataclass
class FrameWork:
    extracted: ExtractedFrame
    depth: np.ndarray
    confidence: np.ndarray | None
    instances: dict[str, InstanceWork]
    output: dict[str, Any]
