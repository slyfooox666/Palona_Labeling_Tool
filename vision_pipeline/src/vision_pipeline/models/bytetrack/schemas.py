"""ByteTrack adapter config."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ByteTrackConfig:
    model_id: str
    track_threshold: float = 0.5
    match_threshold: float = 0.8
    new_track_threshold: float = 0.6
    max_lost_seconds: float = 10.0
    min_track_seconds: float = 1.0

    @classmethod
    def from_mapping(cls, mapping: dict) -> "ByteTrackConfig":
        parameters = mapping.get("parameters", {})
        return cls(
            model_id=mapping.get("model_id", "bytetrack"),
            track_threshold=float(parameters.get("track_threshold", 0.5)),
            match_threshold=float(parameters.get("match_threshold", 0.8)),
            new_track_threshold=float(parameters.get("new_track_threshold", 0.6)),
            max_lost_seconds=float(parameters.get("max_lost_seconds", 10)),
            min_track_seconds=float(parameters.get("min_track_seconds", 1)),
        )
