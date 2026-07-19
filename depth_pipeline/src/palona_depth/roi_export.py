"""Atomic ROI-masked video and normalized Control JSON export.

This module deliberately reuses the streaming SAM3 adapter used by the depth
pipeline.  It never mutates the source video, Control JSON, or ROI/project JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable

import av
import cv2
import numpy as np

from palona_depth.control import iter_control_frames
from palona_depth.models import ControlFrame, ControlTrack
from palona_depth.safety import require_private_json_suffix
from palona_depth.video import probe_video


Progress = Callable[[str], None]
Point = tuple[float, float]


class RoiExportError(ValueError):
    """Raised when an ROI export cannot be completed safely."""


@dataclass(frozen=True)
class RoiExportOptions:
    video_path: Path
    contour_path: Path
    roi_path: Path
    masked_video_path: Path
    filtered_contour_path: Path
    codec: str = "libx264"


@dataclass(frozen=True)
class FilterStatistics:
    frame_count: int
    kept_track_appearances: int
    removed_track_appearances: int


def export_roi(
    options: RoiExportOptions,
    *,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Build both ROI artifacts and atomically install each completed file."""

    report = progress or (lambda _message: None)
    paths = _resolved_paths(options)
    polygon = load_normalized_roi(paths["roi"])
    metadata = probe_video(paths["video"])
    _validate_output_paths(paths)

    masked_target = paths["masked_video"]
    contour_target = paths["filtered_contour"]
    masked_target.parent.mkdir(parents=True, exist_ok=True)
    contour_target.parent.mkdir(parents=True, exist_ok=True)
    _validate_video_container(masked_target)

    masked_temp = _temporary_output_path(masked_target)
    contour_temp = _temporary_output_path(contour_target, suffix=".json.tmp")
    try:
        report("Rendering ROI-masked video")
        frame_count = write_masked_video(
            paths["video"],
            masked_temp,
            polygon,
            codec=options.codec,
        )
        report("Streaming and filtering Control JSON")
        statistics = write_filtered_control_json(
            source_path=paths["contour"],
            output_path=contour_temp,
            output_video_path=masked_target,
            source_video_path=paths["video"],
            roi_source_path=paths["roi"],
            polygon=polygon,
            video_fps=metadata.fps,
            video_width=metadata.width,
            video_height=metadata.height,
        )
        if frame_count <= 0:
            raise RoiExportError("Source video contains no decodable frames")

        # os.replace is atomic on the same filesystem.  Both temporary files
        # live beside their targets, so a failed build never truncates a source
        # or an already-existing output.
        os.replace(masked_temp, masked_target)
        os.replace(contour_temp, contour_target)
        _fsync_directory(masked_target.parent)
        if contour_target.parent != masked_target.parent:
            _fsync_directory(contour_target.parent)
    finally:
        masked_temp.unlink(missing_ok=True)
        contour_temp.unlink(missing_ok=True)

    report("ROI export complete")
    return {
        "masked_video": str(masked_target),
        "filtered_contour": str(contour_target),
        "video_frame_count": frame_count,
        "control_frame_count": statistics.frame_count,
        "kept_track_appearances": statistics.kept_track_appearances,
        "removed_track_appearances": statistics.removed_track_appearances,
        "filter_rule": "keep_track_if_any_contour_area_centroid_is_inside_or_on_roi",
    }


def load_normalized_roi(path: Path) -> tuple[Point, ...]:
    """Load an embedded project ROI, standalone ROI JSON, or sibling roi.json."""

    source = path.expanduser().resolve()
    return _load_normalized_roi(source, visited=set())


def _load_normalized_roi(path: Path, *, visited: set[Path]) -> tuple[Point, ...]:
    if path in visited:
        raise RoiExportError(f"Circular ROI JSON reference: {path}")
    visited.add(path)
    if not path.is_file():
        raise RoiExportError(f"ROI/project JSON does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise RoiExportError(f"Invalid ROI/project JSON {path}: {exc}") from exc

    candidate = _find_polygon(payload)
    if candidate is not None:
        return validate_normalized_polygon(candidate)

    reference = _find_roi_reference(payload)
    if reference is not None:
        referenced_path = Path(reference).expanduser()
        if not referenced_path.is_absolute():
            referenced_path = path.parent / referenced_path
        return _load_normalized_roi(referenced_path.resolve(), visited=visited)

    sibling = path.parent / "roi.json"
    if sibling != path and sibling.is_file():
        return _load_normalized_roi(sibling.resolve(), visited=visited)
    raise RoiExportError(
        f"No normalized ROI polygon found in {path}; expected a polygon/points/roi_polygon field"
    )


def _find_polygon(payload: Any) -> Any | None:
    if _looks_like_polygon(payload):
        return payload
    if not isinstance(payload, dict):
        return None
    for key in (
        "polygon",
        "points",
        "coordinates",
        "roi_polygon",
        "roi_polygon_norm",
        "normalized_polygon",
    ):
        if key in payload and _looks_like_polygon(payload[key]):
            return payload[key]
    roi = payload.get("roi")
    if roi is not None:
        return _find_polygon(roi)
    return None


def _find_roi_reference(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    for key in ("roi_path", "roi_json", "roi_json_path"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    roi = payload.get("roi")
    if isinstance(roi, dict):
        return _find_roi_reference(roi)
    return None


def _looks_like_polygon(value: Any) -> bool:
    return isinstance(value, list) and all(
        isinstance(point, (list, tuple)) and len(point) == 2 for point in value
    )


def validate_normalized_polygon(value: Any) -> tuple[Point, ...]:
    if not isinstance(value, list) or len(value) < 3:
        raise RoiExportError("ROI polygon must contain at least three normalized [x, y] vertices")

    points: list[Point] = []
    for index, point in enumerate(value):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise RoiExportError(f"ROI vertex {index} must be exactly [x, y]")
        if isinstance(point[0], bool) or isinstance(point[1], bool):
            raise RoiExportError(f"ROI vertex {index} coordinates must be numbers")
        try:
            x, y = float(point[0]), float(point[1])
        except (TypeError, ValueError) as exc:
            raise RoiExportError(f"ROI vertex {index} coordinates must be numbers") from exc
        if not math.isfinite(x) or not math.isfinite(y):
            raise RoiExportError(f"ROI vertex {index} coordinates must be finite")
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            raise RoiExportError(f"ROI vertex {index} must be inside normalized range [0, 1]")
        points.append((x, y))

    if len(points) > 3 and points[0] == points[-1]:
        points.pop()
    if len(points) < 3 or len(set(points)) != len(points):
        raise RoiExportError("ROI polygon must contain at least three distinct, non-repeated vertices")
    if _polygon_self_intersects(points):
        raise RoiExportError("ROI polygon must not self-intersect")
    if abs(_signed_polygon_area(points)) <= 1e-10:
        raise RoiExportError("ROI polygon area must be greater than zero")
    return tuple(points)


def write_masked_video(
    source_path: Path,
    output_path: Path,
    polygon: tuple[Point, ...],
    *,
    codec: str,
) -> int:
    """Decode once and encode frames with every pixel outside the ROI set to zero."""

    frame_count = 0
    try:
        with av.open(str(source_path)) as source:
            input_stream = next((stream for stream in source.streams if stream.type == "video"), None)
            if input_stream is None:
                raise RoiExportError(f"No video stream found in {source_path}")
            rate = input_stream.average_rate or input_stream.guessed_rate
            if rate is None or float(rate) <= 0:
                raise RoiExportError(f"Source video has invalid FPS: {source_path}")
            width = int(input_stream.codec_context.width)
            height = int(input_stream.codec_context.height)
            mask = polygon_mask(polygon, width=width, height=height)
            format_name = _container_format(output_path)

            with av.open(str(output_path), mode="w", format=format_name) as destination:
                output_stream = destination.add_stream(codec, rate=rate)
                output_stream.width = width
                output_stream.height = height
                output_stream.pix_fmt = "yuv420p"
                if codec in {"libx264", "h264"}:
                    output_stream.options = {"crf": "18", "preset": "medium"}

                for decoded in source.decode(input_stream):
                    image = decoded.to_ndarray(format="rgb24")
                    if image.shape[:2] != mask.shape:
                        raise RoiExportError(
                            f"Decoded video size changed from {width}x{height} to "
                            f"{image.shape[1]}x{image.shape[0]}"
                        )
                    image[mask == 0] = 0
                    encoded = av.VideoFrame.from_ndarray(image, format="rgb24")
                    encoded.pts = decoded.pts
                    encoded.time_base = decoded.time_base
                    for packet in output_stream.encode(encoded):
                        destination.mux(packet)
                    frame_count += 1
                for packet in output_stream.encode():
                    destination.mux(packet)
        _fsync_file(output_path)
    except RoiExportError:
        raise
    except (av.error.FFmpegError, OSError, ValueError) as exc:
        raise RoiExportError(f"Could not write ROI-masked video {output_path}: {exc}") from exc
    return frame_count


def polygon_mask(polygon: tuple[Point, ...], *, width: int, height: int) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise RoiExportError("Video width and height must be positive")
    vertices = np.asarray(
        [
            [round(x * (width - 1)), round(y * (height - 1))]
            for x, y in polygon
        ],
        dtype=np.int32,
    )
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [vertices], color=255)
    return mask


def write_filtered_control_json(
    *,
    source_path: Path,
    output_path: Path,
    output_video_path: Path,
    source_video_path: Path,
    roi_source_path: Path,
    polygon: tuple[Point, ...],
    video_fps: float,
    video_width: int,
    video_height: int,
) -> FilterStatistics:
    frame_count = 0
    kept_count = 0
    removed_count = 0
    last_frame_index = -1
    last_timestamp = -math.inf

    with output_path.open("w", encoding="utf-8") as handle:
        header = {
            "schema_version": "palona.filtered-contours/v1",
            "video": str(output_video_path),
            "source": {
                "video": str(source_video_path),
                "control_json": str(source_path),
                "roi_json": str(roi_source_path),
            },
            "video_metadata": {
                "width": video_width,
                "height": video_height,
                "fps": video_fps,
            },
            "roi": {
                "coordinate_space": "normalized_xy",
                "polygon": [[x, y] for x, y in polygon],
                "blackout_outside": True,
            },
            "filter_rule": "keep_track_if_any_contour_area_centroid_is_inside_or_on_roi",
        }
        handle.write("{")
        for index, (key, value) in enumerate(header.items()):
            if index:
                handle.write(",")
            handle.write(json.dumps(key))
            handle.write(":")
            json.dump(value, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        handle.write(',"frames":[')

        first_frame = True
        for frame in iter_control_frames(source_path, video_fps=video_fps):
            if frame.frame_index <= last_frame_index:
                raise RoiExportError("Control frames must be strictly ordered by frame_index")
            if frame.timestamp_seconds < last_timestamp:
                raise RoiExportError("Control frames must be ordered by timestamp_seconds")
            last_frame_index = frame.frame_index
            last_timestamp = frame.timestamp_seconds

            tracks = [
                track
                for track in frame.tracks
                if track_centroid_inside_roi(
                    track,
                    polygon,
                    video_width=video_width,
                    video_height=video_height,
                )
            ]
            kept_count += len(tracks)
            removed_count += len(frame.tracks) - len(tracks)
            normalized = _normalized_frame(frame, tracks)
            if not first_frame:
                handle.write(",")
            json.dump(normalized, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
            first_frame = False
            frame_count += 1

        if frame_count == 0:
            raise RoiExportError("Control JSON contains no frames")
        handle.write("]")
        statistics = {
            "frame_count": frame_count,
            "kept_track_appearances": kept_count,
            "removed_track_appearances": removed_count,
        }
        handle.write(',"statistics":')
        json.dump(statistics, handle, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        handle.write("}\n")
        handle.flush()
        os.fsync(handle.fileno())
    return FilterStatistics(frame_count, kept_count, removed_count)


def track_centroid_inside_roi(
    track: ControlTrack,
    polygon: tuple[Point, ...],
    *,
    video_width: int,
    video_height: int,
) -> bool:
    roi = np.asarray(polygon, dtype=np.float32)
    for contour in track.contours_xy:
        centroid = contour_area_centroid(contour)
        if centroid is None:
            continue
        normalized_centroid = (
            centroid[0] / max(1, video_width - 1),
            centroid[1] / max(1, video_height - 1),
        )
        if cv2.pointPolygonTest(roi, normalized_centroid, False) >= 0:
            return True
    return False


def contour_area_centroid(contour: Iterable[Point]) -> Point | None:
    vertices = np.asarray(tuple(contour), dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 2:
        return None
    # OpenCV only interprets float32/int32 arrays as point sets here; a float64
    # (N, 2) array is treated as a tiny raster image and yields a wrong center.
    moments = cv2.moments(vertices.reshape((-1, 1, 2)))
    if abs(float(moments["m00"])) <= 1e-10:
        return None
    return (
        float(moments["m10"] / moments["m00"]),
        float(moments["m01"] / moments["m00"]),
    )


def _normalized_frame(frame: ControlFrame, tracks: list[ControlTrack]) -> dict[str, Any]:
    return {
        "frame_index": frame.frame_index,
        "timestamp_seconds": frame.timestamp_seconds,
        "tracks": [
            {
                "track_id": track.track_id,
                "label": track.label,
                "confidence": track.confidence,
                "contours_xy": [
                    [[float(x), float(y)] for x, y in contour]
                    for contour in track.contours_xy
                ],
                "contours_format": "absolute_xy",
            }
            for track in tracks
        ],
    }


def _resolved_paths(options: RoiExportOptions) -> dict[str, Path]:
    return {
        "video": options.video_path.expanduser().resolve(),
        "contour": options.contour_path.expanduser().resolve(),
        "roi": options.roi_path.expanduser().resolve(),
        "masked_video": options.masked_video_path.expanduser().resolve(),
        "filtered_contour": options.filtered_contour_path.expanduser().resolve(),
    }


def _validate_output_paths(paths: dict[str, Path]) -> None:
    source_paths = {paths["video"], paths["contour"], paths["roi"]}
    for key in ("masked_video", "filtered_contour"):
        output = paths[key]
        if output in source_paths:
            raise RoiExportError(f"Output must not overwrite source file: {output}")
        if output.exists() and output.is_dir():
            raise RoiExportError(f"Output path is a directory: {output}")
    if paths["masked_video"] == paths["filtered_contour"]:
        raise RoiExportError("Masked video and filtered Control JSON outputs must be different files")
    try:
        require_private_json_suffix(
            paths["filtered_contour"],
            suffix=".filtered-contours.json",
            artifact_name="Filtered Control output",
        )
    except ValueError as exc:
        raise RoiExportError(str(exc)) from exc


def _validate_video_container(path: Path) -> None:
    _container_format(path)


def _container_format(path: Path) -> str:
    suffix = path.suffix.lower()
    formats = {".mp4": "mp4", ".m4v": "mp4", ".mov": "mov", ".mkv": "matroska"}
    if suffix not in formats:
        raise RoiExportError("Masked video output must end in .mp4, .m4v, .mov, or .mkv")
    return formats[suffix]


def _temporary_output_path(target: Path, *, suffix: str | None = None) -> Path:
    descriptor, name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.stem}.",
        suffix=suffix if suffix is not None else target.suffix,
    )
    os.close(descriptor)
    return Path(name)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _signed_polygon_area(points: list[Point]) -> float:
    return 0.5 * sum(
        x1 * y2 - x2 * y1
        for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1], strict=True)
    )


def _polygon_self_intersects(points: list[Point]) -> bool:
    edges = list(zip(points, points[1:] + points[:1], strict=True))
    for first_index, first in enumerate(edges):
        for second_index in range(first_index + 1, len(edges)):
            if second_index == first_index + 1 or {first_index, second_index} == {0, len(edges) - 1}:
                continue
            if _segments_intersect(first[0], first[1], edges[second_index][0], edges[second_index][1]):
                return True
    return False


def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    epsilon = 1e-12

    def cross(p: Point, q: Point, r: Point) -> float:
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_segment(p: Point, q: Point, r: Point) -> bool:
        return (
            min(p[0], r[0]) - epsilon <= q[0] <= max(p[0], r[0]) + epsilon
            and min(p[1], r[1]) - epsilon <= q[1] <= max(p[1], r[1]) + epsilon
        )

    abc, abd = cross(a, b, c), cross(a, b, d)
    cda, cdb = cross(c, d, a), cross(c, d, b)
    if ((abc > epsilon and abd < -epsilon) or (abc < -epsilon and abd > epsilon)) and (
        (cda > epsilon and cdb < -epsilon) or (cda < -epsilon and cdb > epsilon)
    ):
        return True
    return (
        (abs(abc) <= epsilon and on_segment(a, c, b))
        or (abs(abd) <= epsilon and on_segment(a, d, b))
        or (abs(cda) <= epsilon and on_segment(c, a, d))
        or (abs(cdb) <= epsilon and on_segment(c, b, d))
    )
