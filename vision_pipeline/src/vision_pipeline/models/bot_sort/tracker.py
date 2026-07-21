"""BoT-SORT person tracker adapter.

This implements the online association core needed by the pipeline: confidence
gating, motion prediction, IoU matching, lightweight appearance matching, track
creation, and lost-track aging. The appearance path uses a color-histogram
fallback today; a neural ReID encoder can be added behind this adapter later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from vision_pipeline.core.schemas import BoundingBox, Detection, Track
from vision_pipeline.models.bot_sort.schemas import BoTSORTConfig

Appearance = tuple[float, ...]


@dataclass
class _TrackState:
    track_id: int
    bbox: BoundingBox
    label: str
    confidence: float
    created_timestamp: float
    last_timestamp: float
    last_seen_timestamp: float
    hits: int = 1
    misses: int = 0
    velocity: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    appearance: Appearance | None = None
    state: Literal["active", "lost", "removed"] = "active"

    def predict(self, timestamp_seconds: float) -> BoundingBox:
        dt = max(0.0, timestamp_seconds - self.last_timestamp)
        vx1, vy1, vx2, vy2 = self.velocity
        return BoundingBox(
            x1=self.bbox.x1 + vx1 * dt,
            y1=self.bbox.y1 + vy1 * dt,
            x2=self.bbox.x2 + vx2 * dt,
            y2=self.bbox.y2 + vy2 * dt,
        )

    def to_track(self, timestamp_seconds: float) -> Track:
        return Track(
            track_id=str(self.track_id),
            bbox=self.predict(timestamp_seconds) if self.state == "lost" else self.bbox,
            label=self.label,
            confidence=self.confidence,
            timestamp_seconds=timestamp_seconds,
            state=self.state,
            metadata={
                "created_timestamp": self.created_timestamp,
                "last_seen_timestamp": self.last_seen_timestamp,
                "hits": self.hits,
                "misses": self.misses,
            },
        )


class BoTSORTTracker:
    """BoT-SORT-style online tracker for people/line tracking."""

    def __init__(self, config: BoTSORTConfig) -> None:
        self.config = config
        self._next_track_id = 1
        self._tracks: dict[int, _TrackState] = {}

    def update(
        self,
        detections: list[Detection],
        timestamp_seconds: float,
    ) -> list[Track]:
        return self.update_with_image(
            detections=detections,
            timestamp_seconds=timestamp_seconds,
            image_path=None,
        )

    def update_with_image(
        self,
        detections: list[Detection],
        timestamp_seconds: float,
        image_path: str | Path | None = None,
    ) -> list[Track]:
        detections = self._attach_appearance(detections, image_path)
        candidate_detections = [
            detection
            for detection in detections
            if detection.confidence >= self.config.track_low_threshold
        ]
        matches, unmatched_track_ids, unmatched_detection_indexes = self._match(
            candidate_detections,
            timestamp_seconds,
        )

        for track_id, detection_index in matches:
            self._update_track(
                self._tracks[track_id],
                candidate_detections[detection_index],
                timestamp_seconds,
            )

        for track_id in unmatched_track_ids:
            self._mark_missed(self._tracks[track_id], timestamp_seconds)

        for detection_index in unmatched_detection_indexes:
            detection = candidate_detections[detection_index]
            if detection.confidence >= self.config.new_track_threshold:
                self._create_track(detection, timestamp_seconds)

        self._drop_removed_tracks()
        return [
            track.to_track(timestamp_seconds)
            for track in sorted(self._tracks.values(), key=lambda item: item.track_id)
            if track.state != "removed"
        ]

    def _match(
        self,
        detections: list[Detection],
        timestamp_seconds: float,
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        track_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if track.state != "removed"
        ]
        if not track_ids or not detections:
            return [], track_ids, list(range(len(detections)))

        candidates: list[tuple[float, int, int]] = []
        for track_id in track_ids:
            track = self._tracks[track_id]
            predicted_bbox = track.predict(timestamp_seconds)
            for detection_index, detection in enumerate(detections):
                if track.label and detection.label and track.label != detection.label:
                    continue
                iou_score = iou(predicted_bbox, detection.bbox)
                center_score = center_proximity(predicted_bbox, detection.bbox)
                appearance_score = appearance_similarity(
                    track.appearance,
                    detection.metadata.get("appearance"),
                )
                score = self._association_score(
                    iou_score=iou_score,
                    center_score=center_score,
                    appearance_score=appearance_score,
                    track=track,
                )
                if self._is_match_candidate(
                    iou_score=iou_score,
                    center_score=center_score,
                    appearance_score=appearance_score,
                    score=score,
                ):
                    candidates.append((score, track_id, detection_index))

        matches: list[tuple[int, int]] = []
        used_track_ids: set[int] = set()
        used_detection_indexes: set[int] = set()
        for _, track_id, detection_index in sorted(candidates, reverse=True):
            if track_id in used_track_ids or detection_index in used_detection_indexes:
                continue
            matches.append((track_id, detection_index))
            used_track_ids.add(track_id)
            used_detection_indexes.add(detection_index)

        unmatched_track_ids = [
            track_id for track_id in track_ids if track_id not in used_track_ids
        ]
        unmatched_detection_indexes = [
            index for index in range(len(detections)) if index not in used_detection_indexes
        ]
        return matches, unmatched_track_ids, unmatched_detection_indexes

    def _create_track(self, detection: Detection, timestamp_seconds: float) -> None:
        track_id = self._next_track_id
        self._next_track_id += 1
        self._tracks[track_id] = _TrackState(
            track_id=track_id,
            bbox=detection.bbox,
            label=detection.label,
            confidence=detection.confidence,
            created_timestamp=timestamp_seconds,
            last_timestamp=timestamp_seconds,
            last_seen_timestamp=timestamp_seconds,
            appearance=normalize_appearance(detection.metadata.get("appearance")),
        )

    def _update_track(
        self,
        track: _TrackState,
        detection: Detection,
        timestamp_seconds: float,
    ) -> None:
        dt = max(1e-6, timestamp_seconds - track.last_timestamp)
        track.velocity = (
            (detection.bbox.x1 - track.bbox.x1) / dt,
            (detection.bbox.y1 - track.bbox.y1) / dt,
            (detection.bbox.x2 - track.bbox.x2) / dt,
            (detection.bbox.y2 - track.bbox.y2) / dt,
        )
        track.bbox = detection.bbox
        track.label = detection.label
        track.confidence = detection.confidence
        track.last_timestamp = timestamp_seconds
        track.last_seen_timestamp = timestamp_seconds
        track.hits += 1
        track.misses = 0
        track.appearance = blend_appearance(
            track.appearance,
            detection.metadata.get("appearance"),
        )
        track.state = "active"

    def _mark_missed(self, track: _TrackState, timestamp_seconds: float) -> None:
        track.bbox = track.predict(timestamp_seconds)
        track.last_timestamp = timestamp_seconds
        track.misses += 1
        if timestamp_seconds - track.last_seen_timestamp > self.config.max_lost_seconds:
            track.state = "removed"
        else:
            track.state = "lost"

    def _drop_removed_tracks(self) -> None:
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if track.state != "removed"
        }

    def _association_score(
        self,
        iou_score: float,
        center_score: float,
        appearance_score: float | None,
        track: _TrackState,
    ) -> float:
        appearance = appearance_score if appearance_score is not None else 0.0
        score = (
            self.config.iou_weight * iou_score
            + self.config.center_weight * center_score
            + self.config.appearance_weight * appearance
        )
        if track.state == "active":
            score += 0.05
        return score

    def _is_match_candidate(
        self,
        iou_score: float,
        center_score: float,
        appearance_score: float | None,
        score: float,
    ) -> bool:
        if iou_score >= self.config.iou_match_threshold:
            return True
        if (
            appearance_score is not None
            and appearance_score >= self.config.appearance_match_threshold
            and center_score >= self.config.center_match_threshold
        ):
            return True
        return score >= self.config.match_threshold

    def _attach_appearance(
        self,
        detections: list[Detection],
        image_path: str | Path | None,
    ) -> list[Detection]:
        if not self.config.reid_enabled:
            return detections

        image = read_image(image_path)
        enriched: list[Detection] = []
        for detection in detections:
            metadata = dict(detection.metadata)
            appearance = metadata.get("appearance")
            if appearance is None and image is not None:
                appearance = color_histogram_embedding(
                    image,
                    detection.bbox,
                    bins=self.config.histogram_bins,
                )
                if appearance is not None:
                    metadata["appearance"] = appearance
            enriched.append(
                Detection(
                    bbox=detection.bbox,
                    label=detection.label,
                    confidence=detection.confidence,
                    model_id=detection.model_id,
                    metadata=metadata,
                )
            )
        return enriched


def iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    x1 = max(box_a.x1, box_b.x1)
    y1 = max(box_a.y1, box_b.y1)
    x2 = min(box_a.x2, box_b.x2)
    y2 = min(box_a.y2, box_b.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    union = box_a.width * box_a.height + box_b.width * box_b.height - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def center_proximity(box_a: BoundingBox, box_b: BoundingBox) -> float:
    ax, ay = box_a.center
    bx, by = box_b.center
    distance = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
    scale = max(box_a.height, box_b.height, box_a.width, box_b.width, 1.0)
    return max(0.0, 1.0 - distance / (scale * 2.5))


def appearance_similarity(
    appearance_a: Appearance | None,
    appearance_b: object,
) -> float | None:
    if appearance_a is None or appearance_b is None:
        return None
    vector_b = tuple(float(value) for value in appearance_b)
    if len(appearance_a) != len(vector_b):
        return None
    dot = sum(a * b for a, b in zip(appearance_a, vector_b, strict=True))
    norm_a = sum(a * a for a in appearance_a) ** 0.5
    norm_b = sum(b * b for b in vector_b) ** 0.5
    if norm_a <= 0 or norm_b <= 0:
        return None
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def blend_appearance(
    current: Appearance | None,
    new: object,
    alpha: float = 0.8,
) -> Appearance | None:
    if new is None:
        return current
    new_vector = tuple(float(value) for value in new)
    if current is None or len(current) != len(new_vector):
        return new_vector
    blended = tuple(
        alpha * current_value + (1.0 - alpha) * new_value
        for current_value, new_value in zip(current, new_vector, strict=True)
    )
    norm = sum(value * value for value in blended) ** 0.5
    if norm <= 0:
        return blended
    return tuple(value / norm for value in blended)


def normalize_appearance(value: object) -> Appearance | None:
    if value is None:
        return None
    return tuple(float(item) for item in value)


def read_image(image_path: str | Path | None) -> object | None:
    if image_path is None:
        return None
    try:
        import cv2
    except ImportError:
        return None
    return cv2.imread(str(image_path))


def color_histogram_embedding(
    image: object,
    bbox: BoundingBox,
    bins: int,
) -> Appearance | None:
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    height, width = image.shape[:2]
    x1 = max(0, min(width - 1, int(round(bbox.x1))))
    y1 = max(0, min(height - 1, int(round(bbox.y1))))
    x2 = max(0, min(width, int(round(bbox.x2))))
    y2 = max(0, min(height, int(round(bbox.y2))))
    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    vector = np.asarray(hist, dtype=float)
    norm = np.linalg.norm(vector)
    if norm <= 0:
        return None
    return tuple((vector / norm).tolist())
