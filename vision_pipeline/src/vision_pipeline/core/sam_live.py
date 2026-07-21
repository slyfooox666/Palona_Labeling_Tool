"""Live SAM windowing, track stitching, and dwell alert state."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from vision_pipeline.core.roi import ROI, roi_from_sequence
from vision_pipeline.core.schemas import BoundingBox, Track
from vision_pipeline.models.sam3.adapter import Sam3FrameResult
from vision_pipeline.utils.video import SampledFrame

SamLiveStrategy = Literal["long_window", "rolling_window", "per_frame"]


@dataclass(frozen=True)
class Sam3LiveConfig:
    """Runtime knobs for SAM-backed live/dwell processing."""

    strategy: SamLiveStrategy = "long_window"
    sample_fps: float = 1.0
    window_seconds: float = 30.0
    stride_seconds: float = 20.0
    dwell_threshold_seconds: float = 120.0
    missing_grace_seconds: float = 20.0
    match_iou_threshold: float = 0.2
    match_distance_threshold_px: float = 120.0
    prompt: tuple[str, ...] = ("basket",)
    roi: ROI | None = None

    @classmethod
    def from_mapping(cls, mapping: dict) -> "Sam3LiveConfig":
        """Build live SAM settings from a use-case YAML subsection."""
        strategy = str(mapping.get("strategy", "long_window")).strip()
        if strategy not in {"long_window", "rolling_window", "per_frame"}:
            raise ValueError(
                "sam3_live.strategy must be long_window, rolling_window, or per_frame"
            )
        sample_fps = float(mapping.get("sample_fps", mapping.get("fps", 1.0)))
        window_seconds = float(mapping.get("window_seconds", 30.0))
        default_stride = window_seconds if strategy == "long_window" else max(
            1.0,
            window_seconds / 2.0,
        )
        prompt_value = mapping.get("prompt", mapping.get("prompts", ("basket",)))
        return cls(
            strategy=strategy,  # type: ignore[arg-type]
            sample_fps=sample_fps,
            window_seconds=window_seconds,
            stride_seconds=float(mapping.get("stride_seconds", default_stride)),
            dwell_threshold_seconds=float(
                mapping.get("dwell_threshold_seconds", 120.0)
            ),
            missing_grace_seconds=float(mapping.get("missing_grace_seconds", 20.0)),
            match_iou_threshold=float(mapping.get("match_iou_threshold", 0.2)),
            match_distance_threshold_px=float(
                mapping.get("match_distance_threshold_px", 120.0)
            ),
            prompt=_normalize_prompts(prompt_value),
            roi=roi_from_sequence(mapping.get("roi")),
        )


@dataclass(frozen=True)
class FrameWindow:
    """A bounded frame window sent to SAM video inference."""

    window_index: int
    frames: list[SampledFrame]
    commit_after_seconds: float

    @property
    def start_seconds(self) -> float:
        return self.frames[0].timestamp_seconds if self.frames else 0.0

    @property
    def end_seconds(self) -> float:
        return self.frames[-1].timestamp_seconds if self.frames else 0.0


@dataclass(frozen=True)
class StitchedFrameResult:
    """Committed global tracks for one frame."""

    timestamp_seconds: float
    image_path: Path
    tracks: list[Track]


@dataclass(frozen=True)
class DwellEvent:
    """A dwell threshold crossing."""

    event_type: str
    timestamp_seconds: float
    global_track_id: str
    dwell_seconds: float
    first_seen_seconds: float
    bbox: BoundingBox
    label: str

    def as_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "timestamp_seconds": self.timestamp_seconds,
            "global_track_id": self.global_track_id,
            "dwell_seconds": self.dwell_seconds,
            "first_seen_seconds": self.first_seen_seconds,
            "bbox_xyxy": self.bbox.as_xyxy(),
            "label": self.label,
        }


@dataclass(frozen=True)
class StitchWindowResult:
    """Globalized output from one SAM window."""

    window_index: int
    committed_frames: list[StitchedFrameResult]
    events: list[DwellEvent]


@dataclass
class _TrackObservation:
    timestamp_seconds: float
    bbox: BoundingBox
    confidence: float


@dataclass
class _GlobalTrackState:
    global_track_id: str
    label: str
    first_seen_seconds: float
    last_seen_seconds: float
    bbox: BoundingBox
    confidence: float
    alerted: bool = False
    state: Literal["active", "lost", "removed"] = "active"
    observations: list[_TrackObservation] = field(default_factory=list)

    def add_observation(self, track: Track) -> None:
        observation = _TrackObservation(
            timestamp_seconds=track.timestamp_seconds,
            bbox=track.bbox,
            confidence=track.confidence,
        )
        self.observations.append(observation)
        self.observations.sort(key=lambda item: item.timestamp_seconds)
        if track.timestamp_seconds >= self.last_seen_seconds:
            self.last_seen_seconds = track.timestamp_seconds
            self.bbox = track.bbox
            self.confidence = track.confidence
            self.label = track.label
            self.state = "active"

    def bbox_near(self, timestamp_seconds: float, max_delta_seconds: float) -> BoundingBox | None:
        if not self.observations:
            return None
        nearest = min(
            self.observations,
            key=lambda item: abs(item.timestamp_seconds - timestamp_seconds),
        )
        if abs(nearest.timestamp_seconds - timestamp_seconds) <= max_delta_seconds:
            return nearest.bbox
        return None


class Sam3TrackStitcher:
    """Map SAM-local window IDs onto long-lived global track IDs."""

    def __init__(self, config: Sam3LiveConfig) -> None:
        self.config = config
        self._next_global_id = 1
        self._tracks: dict[str, _GlobalTrackState] = {}
        self._local_to_global: dict[tuple[int, str], str] = {}

    @property
    def tracks(self) -> dict[str, _GlobalTrackState]:
        return dict(self._tracks)

    def process_window(
        self,
        window_index: int,
        frame_results: Sequence[Sam3FrameResult],
        commit_after_seconds: float,
    ) -> StitchWindowResult:
        """Stitch one SAM result window and emit only newly committed frames."""
        events: list[DwellEvent] = []
        committed_frames: list[StitchedFrameResult] = []
        results_by_time = sorted(frame_results, key=lambda item: item.timestamp_seconds)

        for frame_result in results_by_time:
            timestamp = frame_result.timestamp_seconds
            stitched_tracks: list[Track] = []
            observed_global_ids: set[str] = set()
            used_global_ids: set[str] = set()

            for local_track in sorted(
                frame_result.tracks,
                key=lambda item: (item.bbox.x1, item.bbox.y1, item.track_id),
            ):
                local_key = (window_index, local_track.track_id)
                global_track_id = self._local_to_global.get(local_key)
                if global_track_id is None:
                    global_track_id = self._match_or_create_global_track(
                        local_track,
                        timestamp,
                        used_global_ids,
                    )
                    self._local_to_global[local_key] = global_track_id
                used_global_ids.add(global_track_id)

                global_track = self._commit_or_observe(global_track_id, local_track)
                observed_global_ids.add(global_track_id)

                if timestamp > commit_after_seconds:
                    stitched_track = _as_global_track(local_track, global_track)
                    stitched_tracks.append(stitched_track)
                    event = self._maybe_build_dwell_event(global_track, timestamp)
                    if event is not None:
                        events.append(event)

            if timestamp > commit_after_seconds:
                self._mark_missing_tracks(timestamp, observed_global_ids)
                committed_frames.append(
                    StitchedFrameResult(
                        timestamp_seconds=timestamp,
                        image_path=frame_result.image_path,
                        tracks=stitched_tracks,
                    )
                )

        return StitchWindowResult(
            window_index=window_index,
            committed_frames=committed_frames,
            events=events,
        )

    def _match_or_create_global_track(
        self,
        track: Track,
        timestamp_seconds: float,
        used_global_ids: set[str],
    ) -> str:
        match = self._find_best_match(track, timestamp_seconds, used_global_ids)
        if match is not None:
            return match
        return self._create_global_track(track)

    def _find_best_match(
        self,
        track: Track,
        timestamp_seconds: float,
        used_global_ids: set[str],
    ) -> str | None:
        best_id: str | None = None
        best_score = 0.0
        max_history_delta = max(1.0 / self.config.sample_fps + 0.25, 0.5)

        for global_track_id, candidate in self._tracks.items():
            if global_track_id in used_global_ids:
                continue
            if candidate.state == "removed":
                continue
            if track.label and candidate.label and track.label != candidate.label:
                continue

            reference_bbox = candidate.bbox_near(timestamp_seconds, max_history_delta)
            if reference_bbox is None:
                if timestamp_seconds - candidate.last_seen_seconds > (
                    self.config.missing_grace_seconds
                ):
                    continue
                reference_bbox = candidate.bbox

            score = track_match_score(
                track.bbox,
                reference_bbox,
                iou_threshold=self.config.match_iou_threshold,
                distance_threshold_px=self.config.match_distance_threshold_px,
            )
            if score > best_score:
                best_score = score
                best_id = global_track_id

        return best_id

    def _create_global_track(self, track: Track) -> str:
        global_track_id = str(self._next_global_id)
        self._next_global_id += 1
        self._tracks[global_track_id] = _GlobalTrackState(
            global_track_id=global_track_id,
            label=track.label,
            first_seen_seconds=track.timestamp_seconds,
            last_seen_seconds=track.timestamp_seconds,
            bbox=track.bbox,
            confidence=track.confidence,
            observations=[
                _TrackObservation(
                    timestamp_seconds=track.timestamp_seconds,
                    bbox=track.bbox,
                    confidence=track.confidence,
                )
            ],
        )
        return global_track_id

    def _commit_or_observe(
        self,
        global_track_id: str,
        local_track: Track,
    ) -> _GlobalTrackState:
        global_track = self._tracks[global_track_id]
        global_track.add_observation(local_track)
        return global_track

    def _maybe_build_dwell_event(
        self,
        global_track: _GlobalTrackState,
        timestamp_seconds: float,
    ) -> DwellEvent | None:
        dwell_seconds = timestamp_seconds - global_track.first_seen_seconds
        if global_track.alerted or dwell_seconds < self.config.dwell_threshold_seconds:
            return None

        global_track.alerted = True
        return DwellEvent(
            event_type="dwell_time_exceeded",
            timestamp_seconds=timestamp_seconds,
            global_track_id=global_track.global_track_id,
            dwell_seconds=dwell_seconds,
            first_seen_seconds=global_track.first_seen_seconds,
            bbox=global_track.bbox,
            label=global_track.label,
        )

    def _mark_missing_tracks(
        self,
        timestamp_seconds: float,
        observed_global_ids: set[str],
    ) -> None:
        for global_track_id, global_track in self._tracks.items():
            if global_track_id in observed_global_ids:
                continue
            if timestamp_seconds - global_track.last_seen_seconds > (
                self.config.missing_grace_seconds
            ):
                global_track.state = "removed"
            else:
                global_track.state = "lost"


def build_frame_windows(
    frames: Sequence[SampledFrame],
    config: Sam3LiveConfig,
) -> list[FrameWindow]:
    """Plan file/video frames into SAM windows with optional overlap."""
    ordered_frames = sorted(frames, key=lambda item: item.timestamp_seconds)
    if not ordered_frames:
        return []

    if config.strategy == "per_frame":
        return [
            FrameWindow(
                window_index=index,
                frames=[frame],
                commit_after_seconds=float("-inf"),
            )
            for index, frame in enumerate(ordered_frames)
        ]

    windows: list[FrameWindow] = []
    start_seconds = ordered_frames[0].timestamp_seconds
    last_timestamp = ordered_frames[-1].timestamp_seconds
    last_committed_seconds = float("-inf")
    window_index = 0

    while start_seconds <= last_timestamp:
        end_seconds = start_seconds + config.window_seconds
        window_frames = [
            frame
            for frame in ordered_frames
            if start_seconds <= frame.timestamp_seconds <= end_seconds
        ]
        if window_frames and any(
            frame.timestamp_seconds > last_committed_seconds for frame in window_frames
        ):
            windows.append(
                FrameWindow(
                    window_index=window_index,
                    frames=window_frames,
                    commit_after_seconds=last_committed_seconds,
                )
            )
            last_committed_seconds = window_frames[-1].timestamp_seconds
            window_index += 1

        start_seconds += max(config.stride_seconds, 1.0 / config.sample_fps)
        if start_seconds > last_timestamp and windows:
            break

    return windows


def track_match_score(
    bbox_a: BoundingBox,
    bbox_b: BoundingBox,
    iou_threshold: float,
    distance_threshold_px: float,
) -> float:
    """Return a positive score when two boxes plausibly represent one object."""
    overlap = bbox_iou(bbox_a, bbox_b)
    area_similarity = bbox_area_similarity(bbox_a, bbox_b)
    center_distance = _center_distance(bbox_a, bbox_b)

    if overlap >= iou_threshold:
        return 0.7 * overlap + 0.3 * area_similarity

    if center_distance <= distance_threshold_px:
        distance_score = 1.0 - center_distance / max(distance_threshold_px, 1e-6)
        return 0.55 * distance_score + 0.25 * area_similarity

    return 0.0


def bbox_iou(bbox_a: BoundingBox, bbox_b: BoundingBox) -> float:
    """Intersection-over-union for xyxy boxes."""
    x1 = max(bbox_a.x1, bbox_b.x1)
    y1 = max(bbox_a.y1, bbox_b.y1)
    x2 = min(bbox_a.x2, bbox_b.x2)
    y2 = min(bbox_a.y2, bbox_b.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = bbox_a.width * bbox_a.height + bbox_b.width * bbox_b.height - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def bbox_area_similarity(bbox_a: BoundingBox, bbox_b: BoundingBox) -> float:
    """Return min-area/max-area similarity for two boxes."""
    area_a = bbox_a.width * bbox_a.height
    area_b = bbox_b.width * bbox_b.height
    larger = max(area_a, area_b)
    if larger <= 0:
        return 0.0
    return min(area_a, area_b) / larger


def group_tracks_by_timestamp(tracks: Iterable[Track]) -> dict[float, list[Track]]:
    """Group tracks by timestamp for test and adapter glue code."""
    grouped: dict[float, list[Track]] = defaultdict(list)
    for track in tracks:
        grouped[track.timestamp_seconds].append(track)
    return dict(grouped)


def _as_global_track(local_track: Track, global_track: _GlobalTrackState) -> Track:
    return Track(
        track_id=global_track.global_track_id,
        bbox=local_track.bbox,
        label=local_track.label,
        confidence=local_track.confidence,
        timestamp_seconds=local_track.timestamp_seconds,
        state=global_track.state,
        metadata={
            **local_track.metadata,
            "global_track_id": global_track.global_track_id,
            "local_track_id": local_track.track_id,
            "first_seen_seconds": global_track.first_seen_seconds,
            "dwell_seconds": (
                local_track.timestamp_seconds - global_track.first_seen_seconds
            ),
        },
    )


def _center_distance(bbox_a: BoundingBox, bbox_b: BoundingBox) -> float:
    center_a = bbox_a.center
    center_b = bbox_b.center
    return math.hypot(center_a[0] - center_b[0], center_a[1] - center_b[1])


def _normalize_prompts(value: object) -> tuple[str, ...]:
    if value is None:
        return ("basket",)
    if isinstance(value, str):
        prompts = [value]
    else:
        prompts = list(value)  # type: ignore[arg-type]
    normalized = tuple(str(prompt).strip() for prompt in prompts if str(prompt).strip())
    if not normalized:
        raise ValueError("sam3_live.prompt must contain at least one prompt")
    return normalized
