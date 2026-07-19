"""Strict, streaming merger for single-prompt shared SAM3 manifests."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, BinaryIO, Iterator

import ijson

from palona_depth.control import LFS_POINTER_PREFIX
from palona_depth.safety import require_private_json_suffix


class ControlMergeError(ValueError):
    """Raised when shared SAM3 manifests cannot be merged without ambiguity."""


@dataclass(frozen=True)
class ManifestSource:
    prompt: str
    path: Path


@dataclass(frozen=True)
class ManifestMetadata:
    schema_version: str
    task: str
    input_path: str
    canonical_input_path: Path
    width: int
    height: int
    source_fps: float
    sample_fps: float
    sampled_frames: int
    model_name: str | None
    model_revision: str | None
    device: str | None
    dtype: str | None


@dataclass(frozen=True)
class MergeSummary:
    output_path: Path
    frame_count: int
    track_count: int
    prompts: tuple[str, ...]


def parse_source(value: str) -> ManifestSource:
    prompt, separator, raw_path = value.partition("=")
    prompt = prompt.strip()
    raw_path = raw_path.strip()
    if not separator or not prompt or not raw_path:
        raise argparse.ArgumentTypeError("--source must use PROMPT=/absolute/path/manifest.json")
    return ManifestSource(prompt=prompt, path=Path(raw_path))


def merge_control_manifests(sources: list[ManifestSource], output_path: Path) -> MergeSummary:
    """Merge aligned single-prompt manifests into one normalized Control JSON."""

    if not sources:
        raise ControlMergeError("At least one --source PROMPT=manifest.json is required")
    prompts = tuple(source.prompt.strip() for source in sources)
    if any(not prompt for prompt in prompts):
        raise ControlMergeError("Source prompts must not be empty")
    normalized_prompts = [prompt.casefold() for prompt in prompts]
    if len(normalized_prompts) != len(set(normalized_prompts)):
        raise ControlMergeError("Source prompts must be unique (case-insensitive)")

    resolved_sources = [
        ManifestSource(prompt=source.prompt.strip(), path=source.path.expanduser().resolve())
        for source in sources
    ]
    source_paths = [source.path for source in resolved_sources]
    if len(source_paths) != len(set(source_paths)):
        raise ControlMergeError("Each --source must refer to a different manifest")
    for source in resolved_sources:
        _validate_source_file(source.path)

    output_path = output_path.expanduser().resolve()
    if output_path in set(source_paths):
        raise ControlMergeError("Merged Control output must not overwrite an input manifest")
    require_private_json_suffix(
        output_path,
        suffix=".control.json",
        artifact_name="Merged Control output",
    )

    metadata = [_read_manifest_metadata(source.path) for source in resolved_sources]
    _validate_matching_metadata(resolved_sources, metadata)
    reference = metadata[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path: Path | None = None
    frame_count = 0
    track_count = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_handle:
            temporary_path = Path(output_handle.name)
            header = {
                "schema_version": "palona.control/v1",
                "input_type": "video",
                "video": reference.input_path,
                "input_path": reference.input_path,
                "model": "sam3",
                "created_at": datetime.now(UTC).isoformat(),
                "media": {
                    "width": reference.width,
                    "height": reference.height,
                    "source_fps": reference.source_fps,
                    "sample_fps": reference.sample_fps,
                    "sampled_frames": reference.sampled_frames,
                },
                "sources": [
                    {
                        "prompt_index": index,
                        "prompt": source.prompt,
                        "manifest_path": str(source.path),
                        "schema_version": item.schema_version,
                        "model_name": item.model_name,
                        "model_revision": item.model_revision,
                        "device": item.device,
                        "dtype": item.dtype,
                    }
                    for index, (source, item) in enumerate(zip(resolved_sources, metadata, strict=True))
                ],
            }
            encoded_header = json.dumps(header, ensure_ascii=False, separators=(",", ":"))
            output_handle.write(encoded_header[:-1])
            output_handle.write(',"frames":[')

            with ExitStack() as stack:
                frame_iterators = [
                    _open_frame_iterator(stack.enter_context(source.path.open("rb")), source.path)
                    for source in resolved_sources
                ]
                previous_source_index = -1
                previous_timestamp = -math.inf
                while True:
                    sentinel = object()
                    raw_frames = [next(iterator, sentinel) for iterator in frame_iterators]
                    if all(frame is sentinel for frame in raw_frames):
                        break
                    if any(frame is sentinel for frame in raw_frames):
                        counts = ", ".join(
                            f"{source.prompt}={frame_count + (frame is not sentinel)}+"
                            for source, frame in zip(resolved_sources, raw_frames, strict=True)
                        )
                        raise ControlMergeError(
                            f"Manifest frame counts differ near merged frame {frame_count}: {counts}"
                        )

                    frames = [
                        _normalize_source_frame(
                            frame,
                            source=source,
                            metadata=item,
                            prompt_index=index,
                            ordinal=frame_count,
                        )
                        for index, (frame, source, item) in enumerate(
                            zip(raw_frames, resolved_sources, metadata, strict=True)
                        )
                    ]
                    reference_index = frames[0]["source_frame_index"]
                    reference_timestamp = frames[0]["timestamp_seconds"]
                    for source, frame in zip(resolved_sources[1:], frames[1:], strict=True):
                        if frame["source_frame_index"] != reference_index:
                            raise ControlMergeError(
                                f"Frame {frame_count} source_frame_index mismatch: "
                                f"{resolved_sources[0].prompt}={reference_index}, "
                                f"{source.prompt}={frame['source_frame_index']}"
                            )
                        if frame["timestamp_seconds"] != reference_timestamp:
                            raise ControlMergeError(
                                f"Frame {frame_count} timestamp_seconds mismatch: "
                                f"{resolved_sources[0].prompt}={reference_timestamp}, "
                                f"{source.prompt}={frame['timestamp_seconds']}"
                            )
                    if reference_index <= previous_source_index:
                        raise ControlMergeError("source_frame_index must be strictly increasing")
                    if reference_timestamp <= previous_timestamp:
                        raise ControlMergeError("timestamp_seconds must be strictly increasing")
                    previous_source_index = reference_index
                    previous_timestamp = reference_timestamp

                    tracks = [track for frame in frames for track in frame["tracks"]]
                    track_ids = [track["track_id"] for track in tracks]
                    if len(track_ids) != len(set(track_ids)):
                        raise ControlMergeError(
                            f"Namespaced track IDs collide at source frame {reference_index}"
                        )
                    merged_frame = {
                        "frame_index": reference_index,
                        "source_frame_index": reference_index,
                        "timestamp_seconds": reference_timestamp,
                        "tracks": tracks,
                    }
                    if frame_count:
                        output_handle.write(",")
                    json.dump(merged_frame, output_handle, ensure_ascii=False, separators=(",", ":"))
                    frame_count += 1
                    track_count += len(tracks)

            for source, item in zip(resolved_sources, metadata, strict=True):
                if frame_count != item.sampled_frames:
                    raise ControlMergeError(
                        f"{source.path.name} declares media.sampled_frames={item.sampled_frames} "
                        f"but contains {frame_count} frames"
                    )
            if frame_count == 0:
                raise ControlMergeError("Input manifests contain no frames")
            output_handle.write("]}")
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return MergeSummary(
        output_path=output_path,
        frame_count=frame_count,
        track_count=track_count,
        prompts=prompts,
    )


def _validate_source_file(path: Path) -> None:
    if not path.is_file():
        raise ControlMergeError(f"SAM3 manifest does not exist: {path}")
    with path.open("rb") as handle:
        if handle.read(len(LFS_POINTER_PREFIX)) == LFS_POINTER_PREFIX:
            raise ControlMergeError(f"{path.name} is a Git LFS pointer, not a SAM3 manifest")


def _read_manifest_metadata(path: Path) -> ManifestMetadata:
    scalar_values: dict[str, Any] = {}
    found_frames = False
    try:
        with path.open("rb") as handle:
            for prefix, event, value in ijson.parse(handle):
                if prefix == "frames" and event == "start_array":
                    found_frames = True
                    break
                if event in {"string", "number", "boolean", "null"}:
                    scalar_values[prefix] = value
    except (ijson.JSONError, UnicodeDecodeError, OSError) as exc:
        raise ControlMergeError(f"Invalid SAM3 manifest {path}: {exc}") from exc
    if not found_frames:
        raise ControlMergeError(f"SAM3 manifest {path} is missing a frames array")

    schema_version = _required_string(scalar_values, "schema_version", path)
    task = _required_string(scalar_values, "task", path)
    if schema_version != "ai-model-runtime.sam3/v1" or task != "sam3.track_video":
        raise ControlMergeError(
            f"{path.name} must be an ai-model-runtime.sam3/v1 sam3.track_video manifest"
        )
    input_path = _required_string(scalar_values, "input_path", path)
    width = _positive_int(scalar_values.get("media.width"), f"{path.name} media.width")
    height = _positive_int(scalar_values.get("media.height"), f"{path.name} media.height")
    source_fps = _positive_float(scalar_values.get("media.source_fps"), f"{path.name} media.source_fps")
    sample_fps = _positive_float(scalar_values.get("media.sample_fps"), f"{path.name} media.sample_fps")
    sampled_frames = _positive_int(
        scalar_values.get("media.sampled_frames"),
        f"{path.name} media.sampled_frames",
    )
    return ManifestMetadata(
        schema_version=schema_version,
        task=task,
        input_path=str(Path(input_path).expanduser().resolve()),
        canonical_input_path=Path(input_path).expanduser().resolve(),
        width=width,
        height=height,
        source_fps=source_fps,
        sample_fps=sample_fps,
        sampled_frames=sampled_frames,
        model_name=_optional_string(scalar_values.get("model.model_name")),
        model_revision=_optional_string(scalar_values.get("model.model_revision")),
        device=_optional_string(scalar_values.get("model.device")),
        dtype=_optional_string(scalar_values.get("model.dtype")),
    )


def _validate_matching_metadata(
    sources: list[ManifestSource], metadata: list[ManifestMetadata]
) -> None:
    reference = metadata[0]
    fields = (
        "canonical_input_path",
        "width",
        "height",
        "source_fps",
        "sample_fps",
        "model_name",
        "model_revision",
    )
    for source, item in zip(sources[1:], metadata[1:], strict=True):
        for field in fields:
            expected = getattr(reference, field)
            actual = getattr(item, field)
            if actual != expected:
                raise ControlMergeError(
                    f"Manifest {field} mismatch: {sources[0].prompt}={expected}, "
                    f"{source.prompt}={actual}"
                )


def _open_frame_iterator(handle: BinaryIO, path: Path) -> Iterator[Any]:
    try:
        yield from ijson.items(handle, "frames.item")
    except (ijson.JSONError, UnicodeDecodeError, OSError) as exc:
        raise ControlMergeError(f"Invalid SAM3 frames in {path}: {exc}") from exc


def _normalize_source_frame(
    raw: Any,
    *,
    source: ManifestSource,
    metadata: ManifestMetadata,
    prompt_index: int,
    ordinal: int,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ControlMergeError(f"{source.path.name} frame {ordinal} must be an object")
    if "source_frame_index" not in raw:
        raise ControlMergeError(
            f"{source.path.name} frame {ordinal} is missing source_frame_index; "
            "sampled frame_index is not sufficient for video alignment"
        )
    source_frame_index = _non_negative_int(
        raw["source_frame_index"],
        f"{source.path.name} frame {ordinal} source_frame_index",
    )
    timestamp = _non_negative_float(
        raw.get("timestamp_seconds"),
        f"{source.path.name} frame {ordinal} timestamp_seconds",
    )
    instances = raw.get("instances")
    if not isinstance(instances, list):
        raise ControlMergeError(f"{source.path.name} frame {ordinal} instances must be an array")
    tracks = [
        _normalize_instance(
            instance,
            source=source,
            metadata=metadata,
            prompt_index=prompt_index,
            frame_index=source_frame_index,
            ordinal=instance_index,
        )
        for instance_index, instance in enumerate(instances)
    ]
    original_ids = [track["metadata"]["source_instance_id"] for track in tracks]
    if len(original_ids) != len(set(original_ids)):
        raise ControlMergeError(
            f"{source.path.name} source frame {source_frame_index} contains duplicate instance IDs"
        )
    return {
        "source_frame_index": source_frame_index,
        "timestamp_seconds": timestamp,
        "tracks": tracks,
    }


def _normalize_instance(
    raw: Any,
    *,
    source: ManifestSource,
    metadata: ManifestMetadata,
    prompt_index: int,
    frame_index: int,
    ordinal: int,
) -> dict[str, Any]:
    location = f"{source.path.name} frame {frame_index} instance {ordinal}"
    if not isinstance(raw, dict):
        raise ControlMergeError(f"{location} must be an object")
    raw_id = raw.get("instance_id")
    if raw_id is None or not str(raw_id).strip():
        raise ControlMergeError(f"{location} is missing instance_id")
    source_id = str(raw_id).strip()
    label = str(raw.get("label", "")).strip()
    if not label:
        raise ControlMergeError(f"{location} is missing label")
    if label.casefold() != source.prompt.casefold():
        raise ControlMergeError(
            f"{location} label {label!r} does not match --source prompt {source.prompt!r}"
        )

    score = raw.get("score", raw.get("confidence"))
    confidence = None
    if score is not None:
        confidence = _finite_float(score, f"{location} score")
        if not 0 <= confidence <= 1:
            raise ControlMergeError(f"{location} score must be between 0 and 1")

    bbox = raw.get("bbox_xyxy")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ControlMergeError(f"{location} bbox_xyxy must contain four numbers")
    bbox_xyxy = [_finite_float(value, f"{location} bbox_xyxy") for value in bbox]
    x1, y1, x2, y2 = bbox_xyxy
    if not (0 <= x1 <= x2 <= metadata.width and 0 <= y1 <= y2 <= metadata.height):
        raise ControlMergeError(f"{location} bbox_xyxy is outside media bounds")

    raw_contours = raw.get("contours", raw.get("contours_xy"))
    if not isinstance(raw_contours, list) or not raw_contours:
        raise ControlMergeError(f"{location} contours must be a non-empty array")
    contours: list[list[list[float]]] = []
    for contour_index, raw_contour in enumerate(raw_contours):
        if not isinstance(raw_contour, list) or len(raw_contour) < 3:
            raise ControlMergeError(f"{location} contour {contour_index} needs at least three points")
        contour: list[list[float]] = []
        for point_index, raw_point in enumerate(raw_contour):
            if not isinstance(raw_point, (list, tuple)) or len(raw_point) != 2:
                raise ControlMergeError(
                    f"{location} contour {contour_index} point {point_index} must be [x, y]"
                )
            x = _finite_float(raw_point[0], f"{location} contour x")
            y = _finite_float(raw_point[1], f"{location} contour y")
            if not (0 <= x < metadata.width and 0 <= y < metadata.height):
                raise ControlMergeError(f"{location} contour point is outside media bounds")
            contour.append([x, y])
        contours.append(contour)

    track: dict[str, Any] = {
        "track_id": f"p{prompt_index}:{source_id}",
        "label": source.prompt,
        "bbox_xyxy": bbox_xyxy,
        "contours_xy": contours,
        "contours_format": "absolute_xy",
        "metadata": {
            "source_prompt": source.prompt,
            "source_instance_id": source_id,
            "source_manifest": str(source.path),
        },
    }
    if confidence is not None:
        track["confidence"] = confidence
    return track


def _required_string(values: dict[str, Any], key: str, path: Path) -> str:
    value = values.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ControlMergeError(f"{path.name} is missing non-empty {key}")
    return value.strip()


def _optional_string(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _finite_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ControlMergeError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ControlMergeError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ControlMergeError(f"{label} must be a finite number")
    return result


def _positive_float(value: Any, label: str) -> float:
    result = _finite_float(value, label)
    if result <= 0:
        raise ControlMergeError(f"{label} must be positive")
    return result


def _non_negative_float(value: Any, label: str) -> float:
    result = _finite_float(value, label)
    if result < 0:
        raise ControlMergeError(f"{label} must be non-negative")
    return result


def _positive_int(value: Any, label: str) -> int:
    result = _non_negative_int(value, label)
    if result <= 0:
        raise ControlMergeError(f"{label} must be positive")
    return result


def _non_negative_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ControlMergeError(f"{label} must be an integer")
    try:
        if isinstance(value, Decimal):
            integral = value.to_integral_value()
            if value != integral:
                raise ControlMergeError(f"{label} must be an integer")
            result = int(integral)
        else:
            result = int(value)
            if float(value) != result:
                raise ControlMergeError(f"{label} must be an integer")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ControlMergeError(f"{label} must be an integer") from exc
    if result < 0:
        raise ControlMergeError(f"{label} must be non-negative")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="palona-control-merge",
        description="Merge aligned single-prompt ai-models SAM3 video manifests into Control JSON.",
    )
    parser.add_argument(
        "--source",
        action="append",
        type=parse_source,
        required=True,
        metavar="PROMPT=MANIFEST",
        help="Repeat in prompt order; the first source receives p0:, the second p1:, and so on",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output *.control.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = merge_control_manifests(args.source, args.output)
    except (ControlMergeError, ValueError, OSError) as exc:
        print(f"palona-control-merge: {exc}", file=sys.stderr)
        return 1
    print(
        f"Created {summary.output_path} · {summary.frame_count} aligned frames · "
        f"{summary.track_count} tracks · {', '.join(summary.prompts)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
