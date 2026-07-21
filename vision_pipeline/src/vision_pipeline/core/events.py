"""Small helpers for producing pipeline events."""

from __future__ import annotations

from typing import Any

from vision_pipeline.core.schemas import PipelineEvent


def make_event(
    event_type: str,
    timestamp_seconds: float,
    track_id: str | None = None,
    **payload: Any,
) -> PipelineEvent:
    """Create a typed pipeline event."""
    return PipelineEvent(
        event_type=event_type,
        timestamp_seconds=timestamp_seconds,
        track_id=track_id,
        payload=payload,
    )
