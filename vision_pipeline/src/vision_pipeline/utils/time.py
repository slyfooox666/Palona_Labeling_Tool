"""Time helpers."""

from __future__ import annotations

from datetime import UTC, datetime


def utc_now_iso() -> str:
    """Current UTC timestamp as ISO-8601 text."""
    return datetime.now(UTC).isoformat()
