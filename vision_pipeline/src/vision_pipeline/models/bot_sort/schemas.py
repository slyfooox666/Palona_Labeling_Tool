"""BoT-SORT adapter config."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BoTSORTConfig:
    model_id: str
    track_high_threshold: float = 0.6
    track_low_threshold: float = 0.1
    new_track_threshold: float = 0.7
    match_threshold: float = 0.8
    iou_match_threshold: float = 0.3
    appearance_match_threshold: float = 0.72
    center_match_threshold: float = 0.2
    appearance_weight: float = 0.55
    iou_weight: float = 0.30
    center_weight: float = 0.15
    max_lost_seconds: float = 30.0
    min_track_seconds: float = 2.0
    reid_enabled: bool = True
    reid_model_path: str | None = None
    reid_fallback: str = "color_histogram"
    histogram_bins: int = 16

    @classmethod
    def from_mapping(cls, mapping: dict) -> "BoTSORTConfig":
        parameters = mapping.get("parameters", {})
        reid = mapping.get("reid", {})
        return cls(
            model_id=mapping.get("model_id", "bot_sort"),
            track_high_threshold=float(parameters.get("track_high_threshold", 0.6)),
            track_low_threshold=float(parameters.get("track_low_threshold", 0.1)),
            new_track_threshold=float(parameters.get("new_track_threshold", 0.7)),
            match_threshold=float(parameters.get("match_threshold", 0.8)),
            iou_match_threshold=float(parameters.get("iou_match_threshold", 0.3)),
            appearance_match_threshold=float(parameters.get("appearance_match_threshold", 0.72)),
            center_match_threshold=float(parameters.get("center_match_threshold", 0.2)),
            appearance_weight=float(parameters.get("appearance_weight", 0.55)),
            iou_weight=float(parameters.get("iou_weight", 0.30)),
            center_weight=float(parameters.get("center_weight", 0.15)),
            max_lost_seconds=float(parameters.get("max_lost_seconds", 30)),
            min_track_seconds=float(parameters.get("min_track_seconds", 2)),
            reid_enabled=bool(reid.get("enabled", True)),
            reid_model_path=reid.get("model_path"),
            reid_fallback=str(reid.get("fallback", "color_histogram")),
            histogram_bins=int(reid.get("histogram_bins", 16)),
        )
