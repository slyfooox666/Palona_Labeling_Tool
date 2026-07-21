"""ROI helpers shared by CLIs and live pipelines."""

from __future__ import annotations

from pathlib import Path

from vision_pipeline.core.schemas import BoundingBox, Detection, Track
from vision_pipeline.utils.video import SampledFrame

ROI = tuple[int, int, int, int]


def parse_roi(value: str | None) -> ROI | None:
    """Parse an absolute-pixel ROI string formatted as x1,y1,x2,y2."""
    if value is None:
        return None
    parts = [part.strip() for part in value.replace(" ", ",").split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError("--roi must be formatted as x1,y1,x2,y2")
    try:
        x1, y1, x2, y2 = [int(round(float(part))) for part in parts]
    except ValueError as exc:
        raise ValueError("--roi values must be numeric") from exc
    if x2 <= x1 or y2 <= y1:
        raise ValueError("--roi must satisfy x2 > x1 and y2 > y1")
    return (x1, y1, x2, y2)


def roi_from_sequence(value: object) -> ROI | None:
    """Parse an ROI from a config list/tuple/string."""
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return parse_roi(value)
    parts = list(value)  # type: ignore[arg-type]
    if len(parts) != 4:
        raise ValueError("ROI config must contain exactly four values")
    x1, y1, x2, y2 = [int(round(float(part))) for part in parts]
    if x2 <= x1 or y2 <= y1:
        raise ValueError("ROI config must satisfy x2 > x1 and y2 > y1")
    return (x1, y1, x2, y2)


def crop_frames_to_roi(
    frames: list[SampledFrame],
    output_dir: Path,
    roi: ROI,
) -> list[SampledFrame]:
    """Crop sampled frames to an absolute-pixel ROI."""
    return [
        SampledFrame(
            path=crop_image_to_roi(frame.path, output_dir, roi),
            timestamp_seconds=frame.timestamp_seconds,
        )
        for frame in frames
    ]


def crop_image_to_roi(image_path: Path, output_dir: Path, roi: ROI) -> Path:
    """Crop one image to an absolute-pixel ROI and persist it beside outputs."""
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "SAM3 --roi requires Pillow. Install it with `pip install pillow`."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    x1, y1, x2, y2 = roi
    with Image.open(image_path) as image:
        width, height = image.size
        clamped_roi = (
            min(max(0, x1), width),
            min(max(0, y1), height),
            min(max(0, x2), width),
            min(max(0, y2), height),
        )
        left, top, right, bottom = clamped_roi
        if right <= left or bottom <= top:
            raise ValueError(f"--roi {roi} is outside image bounds {width}x{height}")
        crop = image.crop(clamped_roi)
        crop_path = output_dir / f"{image_path.stem}_roi{image_path.suffix}"
        crop.save(crop_path)
        return crop_path


def offset_detections(detections: list[Detection], roi: ROI) -> list[Detection]:
    """Map crop-local detections back to full-frame coordinates."""
    x_offset, y_offset = roi[0], roi[1]
    return [
        Detection(
            bbox=offset_bbox(detection.bbox, x_offset, y_offset),
            label=detection.label,
            confidence=detection.confidence,
            model_id=detection.model_id,
            metadata={
                **detection.metadata,
                "roi_xyxy": roi,
                "crop_bbox_xyxy": detection.bbox.as_xyxy(),
            },
        )
        for detection in detections
    ]


def offset_tracks(tracks: list[Track], roi: ROI) -> list[Track]:
    """Map crop-local tracks back to full-frame coordinates."""
    x_offset, y_offset = roi[0], roi[1]
    return [
        Track(
            track_id=track.track_id,
            bbox=offset_bbox(track.bbox, x_offset, y_offset),
            label=track.label,
            confidence=track.confidence,
            timestamp_seconds=track.timestamp_seconds,
            state=track.state,
            metadata={
                **track.metadata,
                "roi_xyxy": roi,
                "crop_bbox_xyxy": track.bbox.as_xyxy(),
            },
        )
        for track in tracks
    ]


def offset_bbox(bbox: BoundingBox, x_offset: int, y_offset: int) -> BoundingBox:
    """Map one crop-local bbox back to full-frame coordinates."""
    return BoundingBox(
        x1=bbox.x1 + x_offset,
        y1=bbox.y1 + y_offset,
        x2=bbox.x2 + x_offset,
        y2=bbox.y2 + y_offset,
    )
