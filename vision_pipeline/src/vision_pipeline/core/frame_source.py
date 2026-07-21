"""Frame source configuration primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class FrameSourceConfig:
    """Source definition for RTSP/file inputs."""

    source_type: Literal["rtsp", "file"]
    uri: str
    fps: float


def frame_source_from_mapping(mapping: dict) -> FrameSourceConfig:
    """Build a frame source config from YAML data."""
    return FrameSourceConfig(
        source_type=mapping["type"],
        uri=mapping.get("uri") or mapping.get("uri_env", ""),
        fps=float(mapping.get("fps", 1)),
    )
