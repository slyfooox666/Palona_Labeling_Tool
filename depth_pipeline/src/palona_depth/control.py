"""Streaming SAM3 contour adapter and deterministic temporal sampling."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Iterable, Iterator

import ijson

from palona_depth.models import ControlFrame, ControlTrack, Point


LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec/v1"


class ControlDataError(ValueError):
    """Raised when a Control JSON cannot be normalized safely."""


def sampled_control_frames(
    path: Path,
    *,
    sample_fps: float,
    video_fps: float,
    start_time: float = 0.0,
    end_time: float | None = None,
    max_frames: int | None = None,
) -> list[ControlFrame]:
    if sample_fps <= 0 or not math.isfinite(sample_fps):
        raise ControlDataError("sample_fps must be a finite positive number")
    if video_fps <= 0 or not math.isfinite(video_fps):
        raise ControlDataError("video FPS must be a finite positive number")
    if start_time < 0 or not math.isfinite(start_time):
        raise ControlDataError("start_time must be finite and non-negative")
    if end_time is not None and (not math.isfinite(end_time) or end_time <= start_time):
        raise ControlDataError("end_time must be greater than start_time")
    if max_frames is not None and max_frames <= 0:
        raise ControlDataError("max_frames must be positive")

    frames = iter_control_frames(path, video_fps=video_fps)
    selected: list[ControlFrame] = []
    previous: ControlFrame | None = None
    next_target = start_time
    last_timestamp = -math.inf
    last_selected_index: int | None = None

    for current in frames:
        if current.timestamp_seconds < last_timestamp:
            raise ControlDataError("Control frames must be ordered by timestamp_seconds")
        last_timestamp = current.timestamp_seconds
        if current.timestamp_seconds < start_time:
            previous = current
            continue
        if end_time is not None and previous is not None and previous.timestamp_seconds > end_time:
            break

        while current.timestamp_seconds >= next_target:
            candidates = [
                candidate
                for candidate in (previous, current)
                if candidate is not None and candidate.timestamp_seconds >= start_time
            ]
            chosen = min(candidates, key=lambda item: abs(item.timestamp_seconds - next_target))
            if end_time is not None and chosen.timestamp_seconds > end_time:
                return selected
            if chosen.frame_index != last_selected_index:
                selected.append(chosen)
                last_selected_index = chosen.frame_index
                if max_frames is not None and len(selected) >= max_frames:
                    return selected
            next_target += 1.0 / sample_fps
            if previous is current:
                break
        previous = current

    if not selected and previous is not None and previous.timestamp_seconds >= start_time:
        selected.append(previous)
    if not selected:
        raise ControlDataError("Control JSON contains no usable frames in the requested range")
    return selected


def iter_control_frames(path: Path, *, video_fps: float) -> Iterator[ControlFrame]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise ControlDataError(f"Control JSON does not exist: {path}")
    with path.open("rb") as handle:
        prefix = handle.read(len(LFS_POINTER_PREFIX))
    if prefix == LFS_POINTER_PREFIX:
        raise ControlDataError(
            f"{path.name} is a Git LFS pointer, not Control JSON. Run `git lfs pull` first."
        )

    with path.open("rb") as handle:
        try:
            seen_frame_indexes: set[int] = set()
            previous_frame_index = -1
            for ordinal, raw_frame in enumerate(ijson.items(handle, "frames.item")):
                frame = normalize_frame(raw_frame, ordinal=ordinal, video_fps=video_fps)
                if frame.frame_index in seen_frame_indexes:
                    raise ControlDataError(f"Control JSON contains duplicate frame_index {frame.frame_index}")
                if frame.frame_index <= previous_frame_index:
                    raise ControlDataError("Control frames must be strictly ordered by frame_index")
                seen_frame_indexes.add(frame.frame_index)
                previous_frame_index = frame.frame_index
                yield frame
        except (ijson.JSONError, UnicodeDecodeError, OSError) as exc:
            raise ControlDataError(f"Invalid Control JSON {path}: {exc}") from exc


def normalize_frame(raw: Any, *, ordinal: int, video_fps: float) -> ControlFrame:
    if not isinstance(raw, dict):
        raise ControlDataError(f"Frame {ordinal} must be an object")
    raw_timestamp = raw.get("timestamp_seconds")
    # Shared SAM3 Runtime uses frame_index for the sampled-list ordinal and
    # source_frame_index for the actual source-video frame. Prefer the latter
    # whenever it is available so UI, depth, and exported frame IDs agree.
    raw_index = raw.get("source_frame_index", raw.get("frame_index"))
    if raw_timestamp is None and raw_index is None:
        raise ControlDataError(f"Frame {ordinal} needs frame_index or timestamp_seconds")
    timestamp = float(raw_timestamp) if raw_timestamp is not None else float(raw_index) / video_fps
    frame_index = int(raw_index) if raw_index is not None else int(round(timestamp * video_fps))
    if frame_index < 0 or not math.isfinite(timestamp) or timestamp < 0:
        raise ControlDataError(f"Frame {ordinal} has invalid frame index or timestamp")

    raw_tracks = raw.get("tracks")
    if raw_tracks is None:
        raw_tracks = raw.get("instances", [])
    if not isinstance(raw_tracks, list):
        raise ControlDataError(f"Frame {frame_index} tracks must be an array")
    tracks = tuple(
        normalize_track(item, frame_index=frame_index, ordinal=index)
        for index, item in enumerate(raw_tracks)
    )
    track_ids = [track.track_id for track in tracks]
    if len(track_ids) != len(set(track_ids)):
        raise ControlDataError(f"Frame {frame_index} contains duplicate track IDs")
    return ControlFrame(frame_index=frame_index, timestamp_seconds=timestamp, tracks=tracks)


def normalize_track(raw: Any, *, frame_index: int, ordinal: int) -> ControlTrack:
    if not isinstance(raw, dict):
        raise ControlDataError(f"Frame {frame_index} track {ordinal} must be an object")
    raw_id = raw.get("track_id", raw.get("instance_id", raw.get("object_id")))
    if raw_id is None or not str(raw_id).strip():
        raise ControlDataError(f"Frame {frame_index} track {ordinal} is missing track_id")
    label = str(raw.get("label", raw.get("prompt_label", "unknown"))).strip() or "unknown"
    confidence_value = raw.get("confidence", raw.get("score"))
    confidence = float(confidence_value) if confidence_value is not None else None
    metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
    raw_contours = raw.get("contours_xy", raw.get("contours", metadata.get("contours_xy", [])))
    contours = normalize_contours(raw_contours, frame_index=frame_index, track_id=str(raw_id))
    return ControlTrack(
        track_id=str(raw_id),
        label=label,
        confidence=confidence,
        contours_xy=contours,
    )


def normalize_contours(raw: Any, *, frame_index: int, track_id: str) -> tuple[tuple[Point, ...], ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ControlDataError(f"Frame {frame_index} track {track_id} contours must be an array")
    contours: list[tuple[Point, ...]] = []
    for contour in raw:
        if not isinstance(contour, list):
            continue
        points: list[Point] = []
        for point in contour:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            x, y = float(point[0]), float(point[1])
            if math.isfinite(x) and math.isfinite(y):
                points.append((x, y))
        if len(points) >= 3:
            contours.append(tuple(points))
    return tuple(contours)
