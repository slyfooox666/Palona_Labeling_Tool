"""Instance and person-object spatial/temporal cue extraction."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Iterable

import cv2
import numpy as np

from palona_depth.models import ControlTrack, ExtractedFrame, FrameWork, InstanceWork


def align_relative_depth_maps(
    depth_maps: list[np.ndarray],
    *,
    anchor_masks: list[np.ndarray] | None = None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Remove per-image affine scale/shift drift using fixed-camera background anchors."""
    if not depth_maps:
        raise ValueError("At least one depth map is required for temporal alignment")
    if anchor_masks is not None and len(anchor_masks) != len(depth_maps):
        raise ValueError("Depth map and temporal anchor mask counts do not match")

    shape = depth_maps[0].shape
    if any(depth.shape != shape for depth in depth_maps):
        raise ValueError("All depth maps must share one shape for temporal alignment")
    common_finite = np.logical_and.reduce([np.isfinite(depth) for depth in depth_maps])
    used_full_frame_fallback = anchor_masks is None
    if anchor_masks is not None:
        if any(anchor.shape != shape for anchor in anchor_masks):
            raise ValueError("Temporal anchor mask shape does not match its depth map")
        common_anchor = np.logical_and.reduce(anchor_masks) & common_finite
        if int(common_anchor[::4, ::4].sum()) < 512:
            common_anchor = common_finite
            used_full_frame_fallback = True
    else:
        common_anchor = common_finite

    frame_stats: list[tuple[float, float, float, bool, int]] = []
    for index, depth in enumerate(depth_maps):
        sampled_depth = depth[::4, ::4]
        selected = common_anchor[::4, ::4]
        values = sampled_depth[selected]
        if values.size < 16:
            raise ValueError(f"Depth frame {index} has too few finite pixels for temporal alignment")
        q25, median, q75 = (float(value) for value in np.quantile(values, [0.25, 0.5, 0.75]))
        iqr = q75 - q25
        if not all(math.isfinite(value) for value in (q25, median, q75)) or iqr <= 1e-8:
            raise ValueError(f"Depth frame {index} has an unstable temporal alignment distribution")
        frame_stats.append((median, iqr, q25, used_full_frame_fallback, int(values.size)))

    canonical_median = float(np.median([item[0] for item in frame_stats]))
    canonical_iqr = float(np.median([item[1] for item in frame_stats]))
    aligned: list[np.ndarray] = []
    transforms: list[dict[str, Any]] = []
    for depth, (median, iqr, _q25, used_fallback, anchor_pixels) in zip(
        depth_maps,
        frame_stats,
        strict=True,
    ):
        scale = canonical_iqr / iqr
        shift = canonical_median - scale * median
        aligned.append((depth * scale + shift).astype(np.float32, copy=False))
        transform_quality = math.exp(
            -abs(math.log(scale))
            -0.2 * min(abs(shift) / max(canonical_iqr, 1e-8), 5.0)
        )
        transforms.append(
            {
                "scale": float(scale),
                "shift": float(shift),
                "anchor_median_raw": median,
                "anchor_iqr_raw": iqr,
                "anchor_pixel_count_sampled": anchor_pixels,
                "used_full_frame_fallback": used_fallback,
                "transform_quality": float(np.clip(transform_quality, 0.0, 1.0)),
            }
        )

    # Marginal quantiles alone cannot prove that a fixed camera's *same pixels*
    # stayed stable: a spatial permutation can preserve median/IQR perfectly.
    # Compare every aligned anchor pixel with the per-pixel temporal median so
    # scene motion or illumination/model failures explicitly reduce cue quality.
    selected = common_anchor[::4, ::4]
    aligned_anchor_values = np.stack(
        [depth[::4, ::4][selected].astype(np.float64, copy=False) for depth in aligned],
        axis=0,
    )
    anchor_reference = np.median(aligned_anchor_values, axis=0)
    for values, transform in zip(aligned_anchor_values, transforms, strict=True):
        residual = float(np.median(np.abs(values - anchor_reference)))
        residual_iqr_units = residual / max(canonical_iqr, 1e-8)
        correlation = same_pixel_correlation(values, anchor_reference)
        residual_quality = math.exp(-2.0 * min(residual_iqr_units, 5.0))
        correlation_quality = max(0.0, correlation)
        stability_quality = float(transform.pop("transform_quality")) * math.sqrt(
            residual_quality * correlation_quality
        )
        if bool(transform["used_full_frame_fallback"]):
            # A full-frame anchor may contain moving people or objects. Keep the
            # cue available, but explicitly lower its downstream reliability.
            stability_quality *= 0.75
        transform.update(
            {
                "same_pixel_residual_median_aligned": residual,
                "same_pixel_residual_iqr_units": residual_iqr_units,
                "same_pixel_correlation": correlation,
                "stability_quality": float(np.clip(stability_quality, 0.0, 1.0)),
            }
        )
    return aligned, {
        "method": "per_frame_robust_affine_to_clip_anchor",
        "anchor_region": "common_untracked_background_with_full_frame_fallback",
        "lower_quantile": 0.25,
        "center_quantile": 0.5,
        "upper_quantile": 0.75,
        "canonical_median": canonical_median,
        "canonical_iqr": canonical_iqr,
        "frame_transforms": transforms,
    }


def robust_depth_bounds(depth_maps: Iterable[np.ndarray]) -> tuple[float, float]:
    samples: list[np.ndarray] = []
    for depth in depth_maps:
        values = depth[::4, ::4]
        values = values[np.isfinite(values)]
        if values.size:
            samples.append(values.reshape(-1))
    if not samples:
        raise ValueError("DA3 produced no finite positive depth samples")
    combined = np.concatenate(samples)
    low, high = np.quantile(combined, [0.02, 0.98])
    if not math.isfinite(float(low)) or not math.isfinite(float(high)) or high <= low:
        raise ValueError("Could not compute stable clip-level depth normalization bounds")
    return float(low), float(high)


def build_frame_work(
    extracted: ExtractedFrame,
    depth: np.ndarray,
    confidence: np.ndarray | None,
    *,
    video_width: int,
    video_height: int,
    normalization_low: float,
    normalization_high: float,
    alignment_tolerance_seconds: float,
    person_labels: set[str],
    target_labels: set[str],
    temporal_alignment_quality: float = 1.0,
) -> FrameWork:
    height, width = depth.shape
    instances: dict[str, InstanceWork] = {}
    frame_output: dict[str, Any] = {
        "frame_index": extracted.control.frame_index,
        "timestamp_seconds": extracted.control.timestamp_seconds,
        "decoded_frame_index": extracted.decoded_frame_index,
        "decoded_timestamp_seconds": extracted.decoded_timestamp_seconds,
        "alignment_error_seconds": extracted.alignment_error_seconds,
        "instances": [],
        "pairs": [],
    }
    alignment_quality = max(
        0.0,
        1.0 - extracted.alignment_error_seconds / max(alignment_tolerance_seconds, 1e-9),
    )
    confidence_reference = (
        confidence[np.isfinite(confidence)]
        if confidence is not None
        else np.empty(0, dtype=np.float32)
    )
    for track in extracted.control.tracks:
        mask = contours_to_mask(
            track,
            video_width=video_width,
            video_height=video_height,
            depth_width=width,
            depth_height=height,
        )
        if not mask.any():
            continue
        sample_mask = eroded_or_original(mask)
        valid_mask = sample_mask & np.isfinite(depth)
        valid_count = int(valid_mask.sum())
        mask_count = int(sample_mask.sum())
        if valid_count == 0 or mask_count == 0:
            continue
        values = depth[valid_mask]
        raw_median = float(np.median(values))
        raw_p25, raw_p75 = (float(value) for value in np.quantile(values, [0.25, 0.75]))
        depth_rank = normalize_rank(raw_median, normalization_low, normalization_high)
        depth_iqr = min(1.0, max(0.0, (raw_p75 - raw_p25) / (normalization_high - normalization_low)))
        valid_ratio = valid_count / mask_count
        contour_confidence = clipped_confidence(track.confidence)
        # Missing confidence is unknown, not perfect.  A present but unusable
        # confidence map is penalized further until finite mask values prove it
        # can contribute a within-frame percentile.
        depth_confidence_quality = 0.5 if confidence is None else 0.25
        depth_confidence_median_raw: float | None = None
        if confidence is not None:
            confidence_values = confidence[valid_mask]
            finite_confidence = confidence_values[np.isfinite(confidence_values)]
            if finite_confidence.size and confidence_reference.size:
                depth_confidence_median_raw = float(np.median(finite_confidence))
                # DA3 confidence is not a calibrated [0, 1] probability. Its
                # within-frame percentile is a bounded, model-scale-free cue.
                depth_confidence_quality = float(
                    np.mean(confidence_reference <= depth_confidence_median_raw)
                )
        feature_quality = float(
            np.clip(
                math.sqrt(
                    valid_ratio
                    * contour_confidence
                    * alignment_quality
                    * float(np.clip(temporal_alignment_quality, 0.0, 1.0))
                    * depth_confidence_quality
                ),
                0.0,
                1.0,
            )
        )
        ys, xs = np.nonzero(mask)
        centroid = [float(xs.mean() / width), float(ys.mean() / height)]
        output: dict[str, Any] = {
            "track_id": track.track_id,
            "label": track.label,
            "depth_rank": depth_rank,
            "depth_iqr": depth_iqr,
            "valid_depth_ratio": valid_ratio,
            "centroid_xy_norm": centroid,
            "mask_area_ratio": float(mask.mean()),
            "feature_quality": feature_quality,
            "depth_velocity": 0.0,
        }
        if track.confidence is not None and math.isfinite(track.confidence):
            output["contour_confidence"] = float(track.confidence)
        if depth_confidence_median_raw is not None:
            output["depth_confidence_median_raw"] = depth_confidence_median_raw
            output["depth_confidence_percentile"] = depth_confidence_quality
        instances[track.track_id] = InstanceWork(
            output=output,
            mask=mask,
            raw_depth_median=raw_median,
        )
        frame_output["instances"].append(output)

    person_ids = [
        track_id
        for track_id, item in instances.items()
        if label_matches(str(item.output["label"]), person_labels)
    ]
    target_ids = [
        track_id
        for track_id, item in instances.items()
        if label_matches(str(item.output["label"]), target_labels)
    ]
    diagonal = math.hypot(width, height)
    for source_id in person_ids:
        for target_id in target_ids:
            if source_id == target_id:
                continue
            source = instances[source_id]
            target = instances[target_id]
            local_target_raw, target_visible_ratio = local_target_depth(depth, target.mask, source.mask)
            if local_target_raw is None:
                continue
            target_rank = normalize_rank(local_target_raw, normalization_low, normalization_high)
            source_rank = float(source.output["depth_rank"])
            source_centroid = source.output["centroid_xy_norm"]
            target_centroid = target.output["centroid_xy_norm"]
            centroid_gap = math.hypot(
                float(source_centroid[0]) - float(target_centroid[0]),
                float(source_centroid[1]) - float(target_centroid[1]),
            )
            mask_gap = mask_distance(source.mask, target.mask) / diagonal
            overlap = np.logical_and(source.mask, target.mask).sum()
            overlap_ratio = float(overlap / max(1, min(source.mask.sum(), target.mask.sum())))
            quality = math.sqrt(
                float(source.output["feature_quality"])
                * float(target.output["feature_quality"])
                * target_visible_ratio
            )
            pair = {
                "source_id": source_id,
                "target_id": target_id,
                "source_depth_rank": source_rank,
                "target_local_depth_rank": target_rank,
                "depth_gap_abs": abs(source_rank - target_rank),
                "depth_order": (
                    "source_nearer"
                    if source_rank + 0.01 < target_rank
                    else "target_nearer"
                    if target_rank + 0.01 < source_rank
                    else "similar"
                ),
                "centroid_distance_2d_norm": centroid_gap,
                "mask_gap_2d_norm": mask_gap,
                "mask_overlap_ratio": overlap_ratio,
                "target_visible_ratio": target_visible_ratio,
                "relative_depth_velocity": 0.0,
                "proximity_score": 0.0,
                "proximity_duration_seconds": 0.0,
                "trend": "stable",
                "start_candidate_score": 0.0,
                "end_candidate_score": 0.0,
                "feature_quality": float(np.clip(quality, 0.0, 1.0)),
            }
            frame_output["pairs"].append(pair)

    frame_output["instances"].sort(key=lambda item: str(item["track_id"]))
    frame_output["pairs"].sort(key=lambda item: (str(item["source_id"]), str(item["target_id"])))
    return FrameWork(
        extracted=extracted,
        depth=depth,
        confidence=confidence,
        instances=instances,
        output=frame_output,
    )


def enrich_temporal_features(
    frames: list[FrameWork],
    *,
    start_threshold: float = 0.68,
    end_threshold: float = 0.45,
    start_hold_seconds: float = 0.6,
    end_hold_seconds: float = 0.8,
    max_gap_seconds: float = 0.5,
) -> list[dict[str, Any]]:
    frames.sort(key=lambda item: item.extracted.control.timestamp_seconds)
    previous_instances: dict[str, tuple[float, float]] = {}
    pair_series: dict[tuple[str, str], list[tuple[FrameWork, dict[str, Any]]]] = defaultdict(list)

    for frame in frames:
        timestamp = frame.extracted.control.timestamp_seconds
        for item in frame.output["instances"]:
            track_id = str(item["track_id"])
            previous = previous_instances.get(track_id)
            if (
                previous is not None
                and timestamp > previous[0]
                and timestamp - previous[0] <= max_gap_seconds
            ):
                item["depth_velocity"] = (float(item["depth_rank"]) - previous[1]) / (timestamp - previous[0])
            else:
                item["depth_velocity"] = 0.0
            previous_instances[track_id] = (timestamp, float(item["depth_rank"]))
        for pair in frame.output["pairs"]:
            pair_series[(str(pair["source_id"]), str(pair["target_id"]))].append((frame, pair))

    candidates: list[dict[str, Any]] = []
    candidate_index = 0
    for (source_id, target_id), records in sorted(pair_series.items()):
        previous_time: float | None = None
        previous_depth_gap: float | None = None
        previous_proximity: float | None = None
        smoothed_proximity: float | None = None
        near_since: float | None = None
        far_since: float | None = None
        active_candidate: dict[str, Any] | None = None

        for frame, pair in records:
            timestamp = frame.extracted.control.timestamp_seconds
            if previous_time is not None and timestamp - previous_time > max_gap_seconds:
                if active_candidate is not None:
                    active_candidate["end_time"] = previous_time
                    candidates.append(active_candidate)
                previous_time = None
                previous_depth_gap = None
                previous_proximity = None
                smoothed_proximity = None
                near_since = None
                far_since = None
                active_candidate = None
            depth_gap = float(pair["depth_gap_abs"])
            mask_gap = float(pair["mask_gap_2d_norm"])
            centroid_gap = float(pair["centroid_distance_2d_norm"])
            depth_score = float(np.clip(1.0 - depth_gap / 0.25, 0.0, 1.0))
            mask_score = float(np.clip(1.0 - mask_gap / 0.12, 0.0, 1.0))
            centroid_score = float(np.clip(1.0 - centroid_gap / 0.35, 0.0, 1.0))
            raw_proximity = 0.45 * depth_score + 0.4 * mask_score + 0.15 * centroid_score
            smoothed_proximity = (
                raw_proximity
                if smoothed_proximity is None
                else 0.55 * raw_proximity + 0.45 * smoothed_proximity
            )
            pair["proximity_score"] = smoothed_proximity
            dt = timestamp - previous_time if previous_time is not None else 0.0
            depth_velocity = (
                (depth_gap - previous_depth_gap) / dt
                if previous_depth_gap is not None and dt > 0
                else 0.0
            )
            proximity_velocity = (
                (smoothed_proximity - previous_proximity) / dt
                if previous_proximity is not None and dt > 0
                else 0.0
            )
            pair["relative_depth_velocity"] = depth_velocity
            pair["trend"] = (
                "approaching"
                if proximity_velocity > 0.05
                else "leaving"
                if proximity_velocity < -0.05
                else "stable"
            )
            quality = float(pair["feature_quality"])
            approach_component = float(np.clip(0.5 + proximity_velocity, 0.0, 1.0))
            pair["start_candidate_score"] = quality * (
                0.8 * smoothed_proximity + 0.2 * approach_component
            )
            pair["end_candidate_score"] = quality * (1.0 - smoothed_proximity)

            if smoothed_proximity >= start_threshold and quality >= 0.35:
                if near_since is None:
                    near_since = timestamp
                pair["proximity_duration_seconds"] = timestamp - near_since
                if active_candidate is None and pair["proximity_duration_seconds"] >= start_hold_seconds:
                    active_candidate = {
                        "candidate_id": f"d{candidate_index}",
                        "source_id": source_id,
                        "target_id": target_id,
                        "start_time": near_since,
                        "end_time": None,
                        "peak_score": float(pair["start_candidate_score"]),
                        "quality": quality,
                    }
                    candidate_index += 1
                    far_since = None
            elif active_candidate is None:
                near_since = None
                pair["proximity_duration_seconds"] = 0.0

            if near_since is not None:
                pair["proximity_duration_seconds"] = timestamp - near_since

            if active_candidate is not None:
                active_candidate["peak_score"] = max(
                    float(active_candidate["peak_score"]),
                    float(pair["start_candidate_score"]),
                )
                active_candidate["quality"] = min(float(active_candidate["quality"]), quality)
                if smoothed_proximity < end_threshold:
                    if far_since is None:
                        far_since = timestamp
                    if timestamp - far_since >= end_hold_seconds:
                        active_candidate["end_time"] = far_since
                        candidates.append(active_candidate)
                        active_candidate = None
                        near_since = None
                        far_since = None
                else:
                    far_since = None

            previous_time = timestamp
            previous_depth_gap = depth_gap
            previous_proximity = smoothed_proximity

        if active_candidate is not None:
            candidates.append(active_candidate)

    return candidates


def contours_to_mask(
    track: ControlTrack,
    *,
    video_width: int,
    video_height: int,
    depth_width: int,
    depth_height: int,
) -> np.ndarray:
    mask = np.zeros((depth_height, depth_width), dtype=np.uint8)
    scale_x = depth_width / video_width
    scale_y = depth_height / video_height
    polygons = []
    for contour in track.contours_xy:
        points = np.asarray(
            [
                [
                    np.clip(round(x * scale_x), 0, depth_width - 1),
                    np.clip(round(y * scale_y), 0, depth_height - 1),
                ]
                for x, y in contour
            ],
            dtype=np.int32,
        )
        if points.shape[0] >= 3:
            polygons.append(points)
    if polygons:
        cv2.fillPoly(mask, polygons, 1)
    return mask.astype(bool)


def eroded_or_original(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), dtype=np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return eroded if int(eroded.sum()) >= 16 else mask


def local_target_depth(
    depth: np.ndarray,
    target_mask: np.ndarray,
    source_mask: np.ndarray,
) -> tuple[float | None, float]:
    valid_target = target_mask & np.isfinite(depth)
    visible_target = valid_target & ~source_mask
    target_y, target_x = np.nonzero(visible_target)
    source_y, source_x = np.nonzero(source_mask)
    full_target_count = int(valid_target.sum())
    if target_x.size < 16 or source_x.size == 0 or full_target_count == 0:
        return None, 0.0
    center_x, center_y = float(source_x.mean()), float(source_y.mean())
    distances = (target_x - center_x) ** 2 + (target_y - center_y) ** 2
    count = min(target_x.size, max(16, int(math.ceil(target_x.size * 0.1))))
    indexes = np.argpartition(distances, count - 1)[:count]
    return (
        float(np.median(depth[target_y[indexes], target_x[indexes]])),
        float(target_x.size / full_target_count),
    )


def mask_distance(left: np.ndarray, right: np.ndarray) -> float:
    if np.logical_and(left, right).any():
        return 0.0
    distance = cv2.distanceTransform((~left).astype(np.uint8), cv2.DIST_L2, 3)
    values = distance[right]
    return float(values.min()) if values.size else float(math.hypot(*left.shape))


def normalize_rank(value: float, low: float, high: float) -> float:
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def clipped_confidence(value: float | None) -> float:
    if value is None or not math.isfinite(value):
        return 0.5
    return float(np.clip(value, 0.0, 1.0))


def same_pixel_correlation(values: np.ndarray, reference: np.ndarray) -> float:
    """Finite Pearson correlation with deterministic degenerate handling."""

    centered_values = values - float(np.mean(values))
    centered_reference = reference - float(np.mean(reference))
    denominator = float(
        np.linalg.norm(centered_values) * np.linalg.norm(centered_reference)
    )
    if denominator <= 1e-12:
        return 1.0 if np.allclose(values, reference, rtol=1e-6, atol=1e-8) else 0.0
    correlation = float(np.dot(centered_values, centered_reference) / denominator)
    return float(np.clip(correlation, -1.0, 1.0))


def label_matches(label: str, configured: set[str]) -> bool:
    normalized = label.casefold().strip()
    return normalized in configured or any(token in normalized for token in configured)
