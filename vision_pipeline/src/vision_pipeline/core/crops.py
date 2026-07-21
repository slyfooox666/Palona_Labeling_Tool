"""Crop geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass

from vision_pipeline.core.schemas import BoundingBox


@dataclass(frozen=True)
class CropPolicy:
    """Controls tight/context crop expansion."""

    tight_padding: float = 0.05
    context_padding: float = 0.35


def expand_bbox(
    bbox: BoundingBox,
    frame_width: int,
    frame_height: int,
    padding_ratio: float,
) -> BoundingBox:
    """Expand a bbox by a ratio of its own width/height and clamp to frame bounds."""
    pad_x = bbox.width * padding_ratio
    pad_y = bbox.height * padding_ratio
    return BoundingBox(
        x1=max(0.0, bbox.x1 - pad_x),
        y1=max(0.0, bbox.y1 - pad_y),
        x2=min(float(frame_width), bbox.x2 + pad_x),
        y2=min(float(frame_height), bbox.y2 + pad_y),
    )


def make_tight_and_context_boxes(
    bbox: BoundingBox,
    frame_width: int,
    frame_height: int,
    policy: CropPolicy,
) -> tuple[BoundingBox, BoundingBox]:
    """Return tight and wider context boxes for one observation."""
    tight = expand_bbox(bbox, frame_width, frame_height, policy.tight_padding)
    context = expand_bbox(bbox, frame_width, frame_height, policy.context_padding)
    return tight, context
