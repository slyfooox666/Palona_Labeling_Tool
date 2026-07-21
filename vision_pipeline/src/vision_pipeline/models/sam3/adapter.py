"""SAM3 promptable segmentation and tracking adapter."""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vision_pipeline.core.schemas import BoundingBox, Detection, Track
from vision_pipeline.models.sam3.schemas import SAM3Config
from vision_pipeline.utils.video import SampledFrame


@dataclass(frozen=True)
class Sam3FrameResult:
    """SAM3 tracks for a sampled video frame."""

    timestamp_seconds: float
    image_path: Path
    tracks: list[Track]
    frame_index: int | None = None


class SAM3ModelAdapter:
    """Adapter boundary for SAM3 image segmentation and video tracking."""

    def __init__(self, config: SAM3Config) -> None:
        self.config = config
        self._image_model: Any | None = None
        self._image_processor: Any | None = None
        self._video_model: Any | None = None
        self._video_processor: Any | None = None
        self._native_video_predictor: Any | None = None

    def detect_image_path(
        self,
        image_path: str | Path,
        prompt: str | Sequence[str] | None = None,
        timestamp_seconds: float = 0.0,
    ) -> list[Detection]:
        """Run SAM3 image segmentation using one or more text prompts."""
        model, processor, torch = self._load_image_model()
        image = _read_pil_image(image_path)
        prompts = self._resolve_prompts(prompt)

        detections: list[Detection] = []
        for text_prompt in prompts:
            inputs = processor(
                images=image,
                text=text_prompt,
                return_tensors="pt",
            )
            inputs = _move_inputs_to_model(inputs, model, self.config.device)

            with torch.no_grad():
                outputs = model(**inputs)

            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=self.config.confidence_threshold,
                mask_threshold=self.config.mask_threshold,
                target_sizes=_original_sizes(inputs),
            )
            if not results:
                continue
            detections.extend(
                self._detections_from_result(
                    results[0],
                    label=text_prompt,
                    timestamp_seconds=timestamp_seconds,
                )
            )
        return detections

    def track_video_frames(
        self,
        frames: list[SampledFrame],
        prompt: str | Sequence[str] | None = None,
    ) -> list[Sam3FrameResult]:
        """Run SAM3 video tracking over sampled frames."""
        if not frames:
            return []

        model, processor, torch = self._load_video_model()
        prompts = self._resolve_prompts(prompt)
        images = [_read_pil_image(frame.path) for frame in frames]
        dtype = _torch_dtype(torch, self.config.dtype)
        session_kwargs: dict[str, Any] = {
            "video": images,
            "inference_device": _model_device(model, self.config.device),
            "processing_device": self.config.processing_device,
            "video_storage_device": self.config.video_storage_device,
            "max_vision_features_cache_size": self.config.max_vision_features_cache_size,
        }
        if self.config.inference_state_device:
            session_kwargs["inference_state_device"] = self.config.inference_state_device
        if dtype is not None:
            session_kwargs["dtype"] = dtype

        inference_session = processor.init_video_session(**session_kwargs)
        inference_session = processor.add_text_prompt(
            inference_session=inference_session,
            text=prompts[0] if len(prompts) == 1 else list(prompts),
        )

        max_frame_num_to_track = self.config.max_frame_num_to_track
        if max_frame_num_to_track is None:
            max_frame_num_to_track = max(0, len(frames) - 1)

        outputs_by_index: dict[int, dict[str, Any]] = {}
        for model_outputs in model.propagate_in_video_iterator(
            inference_session=inference_session,
            max_frame_num_to_track=max_frame_num_to_track,
            show_progress_bar=self.config.show_progress_bar,
        ):
            frame_index = _frame_index(model_outputs, fallback=len(outputs_by_index))
            outputs_by_index[frame_index] = processor.postprocess_outputs(
                inference_session,
                model_outputs,
            )

        frame_results: list[Sam3FrameResult] = []
        default_label = prompts[0] if len(prompts) == 1 else "sam3_object"
        for index, frame in enumerate(frames):
            processed_outputs = outputs_by_index.get(index, {})
            tracks = self._tracks_from_result(
                processed_outputs,
                timestamp_seconds=frame.timestamp_seconds,
                frame_index=index,
                default_label=default_label,
            )
            frame_results.append(
                Sam3FrameResult(
                    timestamp_seconds=frame.timestamp_seconds,
                    image_path=frame.path,
                    tracks=tracks,
                    frame_index=index,
                )
            )

        return frame_results

    def track_video_path(
        self,
        video_path: str | Path,
        prompt: str | Sequence[str] | None = None,
        fps: float | None = None,
        prompt_frame_index: int = 0,
        max_frames: int | None = None,
    ) -> list[Sam3FrameResult]:
        """Run native SAM3 video tracking on a video file path.

        This path keeps the source video intact instead of pre-sampling it into
        individual image files. It uses Meta's native SAM3 predictor API, which
        accepts an MP4/video path or a JPEG frame folder as its session resource.
        """
        predictor = self._load_native_video_predictor()
        prompts = self._resolve_prompts(prompt)
        video_resource = str(video_path)

        outputs_by_index: dict[int, object] = {}
        for prompt_index, text_prompt in enumerate(prompts):
            prompt_outputs_by_index = self._track_native_video_single_prompt(
                predictor=predictor,
                video_resource=video_resource,
                text_prompt=text_prompt,
                object_id_namespace=f"p{prompt_index}",
                prompt_frame_index=prompt_frame_index,
                max_frames=max_frames,
            )
            for frame_index, prompt_output in prompt_outputs_by_index.items():
                outputs_by_index[frame_index] = _merge_native_outputs(
                    outputs_by_index.get(frame_index),
                    prompt_output,
                )

        default_label = prompts[0] if len(prompts) == 1 else "sam3_object"
        frame_results: list[Sam3FrameResult] = []
        for frame_index in sorted(outputs_by_index):
            timestamp_seconds = (
                frame_index / fps if fps is not None and fps > 0 else float(frame_index)
            )
            tracks = self._tracks_from_native_output(
                outputs_by_index[frame_index],
                timestamp_seconds=timestamp_seconds,
                frame_index=frame_index,
                default_label=default_label,
            )
            frame_results.append(
                Sam3FrameResult(
                    timestamp_seconds=timestamp_seconds,
                    image_path=Path(video_resource),
                    tracks=tracks,
                    frame_index=frame_index,
                )
            )

        return frame_results

    def _track_native_video_single_prompt(
        self,
        *,
        predictor: Any,
        video_resource: str,
        text_prompt: str,
        object_id_namespace: str,
        prompt_frame_index: int,
        max_frames: int | None,
    ) -> dict[int, object]:
        response = predictor.handle_request(
            request={
                "type": "start_session",
                "resource_path": video_resource,
            }
        )
        session_id = str(response["session_id"])

        try:
            outputs_by_index: dict[int, object] = {}
            prompt_response = predictor.handle_request(
                request={
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": prompt_frame_index,
                    "text": text_prompt,
                }
            )
            prompt_frame = int(prompt_response.get("frame_index", prompt_frame_index))
            if "outputs" in prompt_response and (
                max_frames is None or prompt_frame < max_frames
            ):
                prompt_outputs = _native_output_with_prompt_mapping(
                    prompt_response["outputs"],
                    text_prompt,
                    object_id_namespace,
                )
                outputs_by_index[prompt_frame] = _merge_native_outputs(
                    outputs_by_index.get(prompt_frame),
                    prompt_outputs,
                )

            for stream_response in predictor.handle_stream_request(
                request={
                    "type": "propagate_in_video",
                    "session_id": session_id,
                }
            ):
                frame_index = int(
                    stream_response.get(
                        "frame_index",
                        len(outputs_by_index),
                    )
                )
                if max_frames is not None and frame_index >= max_frames:
                    break
                stream_outputs = _native_output_with_prompt_mapping(
                    stream_response.get("outputs", stream_response),
                    text_prompt,
                    object_id_namespace,
                )
                outputs_by_index[frame_index] = _merge_native_outputs(
                    outputs_by_index.get(frame_index),
                    stream_outputs,
                )
            return outputs_by_index
        finally:
            try:
                predictor.handle_request(
                    request={
                        "type": "close_session",
                        "session_id": session_id,
                    }
                )
            except Exception:
                pass

    def _load_image_model(self) -> tuple[Any, Any, Any]:
        if self._image_model is not None and self._image_processor is not None:
            return self._image_model, self._image_processor, _import_torch()

        torch = _import_torch()
        try:
            from transformers import Sam3Model, Sam3Processor
        except ImportError as exc:
            raise _dependency_error() from exc

        self._image_model = _load_transformers_model(
            Sam3Model,
            self.config.model_name,
            torch,
            self.config.device,
            self.config.device_map,
            self.config.dtype,
        )
        self._image_processor = Sam3Processor.from_pretrained(self.config.model_name)
        return self._image_model, self._image_processor, torch

    def _load_video_model(self) -> tuple[Any, Any, Any]:
        if self._video_model is not None and self._video_processor is not None:
            return self._video_model, self._video_processor, _import_torch()

        torch = _import_torch()
        try:
            from transformers import Sam3VideoModel, Sam3VideoProcessor
        except ImportError as exc:
            raise _dependency_error() from exc

        self._video_model = _load_transformers_model(
            Sam3VideoModel,
            self.config.model_name,
            torch,
            self.config.device,
            self.config.device_map,
            self.config.dtype,
        )
        self._video_processor = Sam3VideoProcessor.from_pretrained(
            self.config.model_name
        )
        return self._video_model, self._video_processor, torch

    def _load_native_video_predictor(self) -> Any:
        if self._native_video_predictor is not None:
            return self._native_video_predictor

        try:
            from sam3.model_builder import build_sam3_predictor
        except ImportError:
            try:
                from sam3.model_builder import build_sam3_video_predictor
            except ImportError as exc:
                raise _native_dependency_error() from exc

            kwargs: dict[str, Any] = {
                "checkpoint_path": self.config.checkpoint_path,
                "compile": self.config.compile_model,
            }
            if self.config.gpus_to_use is not None:
                kwargs["gpus_to_use"] = self.config.gpus_to_use
            self._native_video_predictor = _call_with_supported_kwargs(
                build_sam3_video_predictor,
                kwargs,
            )
            return self._native_video_predictor

        kwargs = {
            "version": self.config.native_version,
            "checkpoint_path": self.config.checkpoint_path,
            "compile": self.config.compile_model,
            "warm_up": self.config.warm_up,
            "max_num_objects": self.config.max_num_objects,
            "multiplex_count": self.config.multiplex_count,
            "async_loading_frames": self.config.async_loading_frames,
        }
        if self.config.gpus_to_use is not None:
            kwargs["gpus_to_use"] = self.config.gpus_to_use

        self._native_video_predictor = _call_with_supported_kwargs(
            build_sam3_predictor,
            kwargs,
        )
        return self._native_video_predictor

    def reset_native_video_predictor(self) -> None:
        self._native_video_predictor = None

    def _resolve_prompts(self, prompt: str | Sequence[str] | None) -> tuple[str, ...]:
        if prompt is None:
            return self.config.prompts
        if isinstance(prompt, str):
            prompts = (prompt,)
        else:
            prompts = tuple(prompt)
        normalized = tuple(
            concept
            for text in prompts
            for concept in _split_prompt_text(text)
        )
        if not normalized:
            raise ValueError("SAM3 requires at least one non-empty text prompt.")
        return normalized

    def _detections_from_result(
        self,
        result: object,
        label: str,
        timestamp_seconds: float,
    ) -> list[Detection]:
        boxes = _as_list(_result_value(result, "boxes", "pred_boxes"))
        scores = _as_list(_result_value(result, "scores", "confidence"))
        masks = (
            _as_mask_list(_result_value(result, "masks"))
            if self.config.include_masks
            else []
        )

        detections: list[Detection] = []
        for index, raw_box in enumerate(boxes):
            box = _parse_box(raw_box)
            if box is None:
                continue
            confidence = _score_at(scores, index)
            if confidence < self.config.confidence_threshold:
                continue
            metadata: dict[str, Any] = {
                "prompt": label,
                "timestamp_seconds": timestamp_seconds,
                "model_name": self.config.model_name,
            }
            if self.config.include_masks and index < len(masks):
                metadata["mask"] = masks[index]
            detections.append(
                Detection(
                    bbox=box,
                    label=label,
                    confidence=confidence,
                    model_id=self.config.model_id,
                    metadata=metadata,
                )
            )
        return detections

    def _tracks_from_result(
        self,
        result: object,
        timestamp_seconds: float,
        frame_index: int,
        default_label: str,
    ) -> list[Track]:
        boxes = _as_list(_result_value(result, "boxes", "pred_boxes"))
        scores = _as_list(_result_value(result, "scores", "confidence"))
        object_ids = _as_list(_result_value(result, "object_ids", "obj_ids", "ids"))
        masks = (
            _as_mask_list(_result_value(result, "masks"))
            if self.config.include_masks
            else []
        )

        tracks: list[Track] = []
        for index, raw_box in enumerate(boxes):
            box = _parse_box(raw_box)
            if box is None:
                continue
            confidence = _score_at(scores, index)
            if confidence < self.config.confidence_threshold:
                continue
            object_id = _object_id_at(object_ids, index)
            label = _prompt_for_object_id(result, object_id, default_label)
            metadata: dict[str, Any] = {
                "sam3_object_id": object_id,
                "prompt": label,
                "frame_index": frame_index,
                "model_name": self.config.model_name,
            }
            if self.config.include_masks and index < len(masks):
                metadata["mask"] = masks[index]
            tracks.append(
                Track(
                    track_id=object_id,
                    bbox=box,
                    label=label,
                    confidence=confidence,
                    timestamp_seconds=timestamp_seconds,
                    metadata=metadata,
                )
            )
        return tracks

    def _tracks_from_native_output(
        self,
        result: object,
        timestamp_seconds: float,
        frame_index: int,
        default_label: str,
    ) -> list[Track]:
        boxes = _as_list(
            _result_value(
                result,
                "boxes",
                "pred_boxes",
                "out_boxes",
                "out_obj_boxes",
            )
        )
        scores = _as_list(
            _result_value(
                result,
                "scores",
                "confidence",
                "confidences",
                "object_scores",
                "out_scores",
            )
        )
        object_ids = _as_list(
            _result_value(
                result,
                "out_obj_ids",
                "object_ids",
                "obj_ids",
                "ids",
            )
        )
        masks = _as_mask_list(
            _result_value(
                result,
                "out_binary_masks",
                "binary_masks",
                "masks",
                "pred_masks",
                "out_mask_logits",
                "mask_logits",
            )
        )

        item_count = max(len(boxes), len(masks), len(object_ids))
        tracks: list[Track] = []
        for index in range(item_count):
            raw_box = boxes[index] if index < len(boxes) else None
            mask = masks[index] if index < len(masks) else None
            box = _parse_box(raw_box) if raw_box is not None else _bbox_from_mask(mask)
            if box is None:
                continue

            confidence = _score_at(scores, index) if scores else 1.0
            if confidence < self.config.confidence_threshold:
                continue

            object_id = _object_id_at(object_ids, index)
            label = _prompt_for_object_id(result, object_id, default_label)
            metadata: dict[str, Any] = {
                "sam3_object_id": object_id,
                "prompt": label,
                "frame_index": frame_index,
                "model_name": self.config.model_name,
                "video_mode": "whole",
            }
            if mask is not None:
                metadata["mask"] = mask
            tracks.append(
                Track(
                    track_id=object_id,
                    bbox=box,
                    label=label,
                    confidence=confidence,
                    timestamp_seconds=timestamp_seconds,
                    metadata=metadata,
                )
            )
        return tracks


def _load_transformers_model(
    model_class: object,
    model_name: str,
    torch: object,
    device: str,
    device_map: str | None,
    dtype_name: str,
) -> object:
    dtype = _torch_dtype(torch, dtype_name)
    model_kwargs: dict[str, object] = {}
    if device_map:
        model_kwargs["device_map"] = device_map
    if dtype is not None:
        model_kwargs["dtype"] = dtype

    try:
        model = model_class.from_pretrained(model_name, **model_kwargs)
    except TypeError:
        if "dtype" not in model_kwargs:
            raise
        model_kwargs["torch_dtype"] = model_kwargs.pop("dtype")
        model = model_class.from_pretrained(model_name, **model_kwargs)

    if not device_map:
        if dtype is not None:
            model = model.to(device, dtype=dtype)
        else:
            model = model.to(device)
    if hasattr(model, "eval"):
        model.eval()
    return model


def _call_with_supported_kwargs(factory: object, kwargs: dict[str, Any]) -> object:
    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):
        return factory(**kwargs)  # type: ignore[misc]

    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return factory(**kwargs)  # type: ignore[misc]

    supported_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters and value is not None
    }
    return factory(**supported_kwargs)  # type: ignore[misc]


def _split_prompt_text(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in str(text).split(",") if part.strip())


def _native_output_with_prompt_mapping(
    output: object,
    prompt: str,
    object_id_namespace: str,
) -> object:
    if not isinstance(output, dict):
        return output
    mapped = dict(output)

    object_ids = _as_list(
        _result_value(mapped, "out_obj_ids", "object_ids", "obj_ids", "ids")
    )
    if not object_ids:
        item_count = max(
            len(
                _as_list(
                    _result_value(
                        mapped,
                        "boxes",
                        "pred_boxes",
                        "out_boxes",
                        "out_obj_boxes",
                    )
                )
            ),
            len(
                _as_mask_list(
                    _result_value(
                        mapped,
                        "out_binary_masks",
                        "binary_masks",
                        "masks",
                        "pred_masks",
                        "out_mask_logits",
                        "mask_logits",
                    )
                )
            ),
        )
        object_ids = list(range(item_count))

    namespaced_ids = [
        _namespace_object_id(object_id, object_id_namespace)
        for object_id in object_ids
    ]
    if namespaced_ids:
        id_key_found = False
        for key in ("out_obj_ids", "object_ids", "obj_ids", "ids"):
            if key in mapped:
                mapped[key] = namespaced_ids
                id_key_found = True
        if not id_key_found:
            mapped["out_obj_ids"] = namespaced_ids

    prompt_to_obj_ids = {prompt: namespaced_ids}
    mapped["prompt_to_obj_ids"] = prompt_to_obj_ids
    return mapped


def _namespace_object_id(object_id: object, namespace: str) -> str:
    raw_id = str(_to_builtin(object_id))
    return f"{namespace}:{raw_id}" if namespace else raw_id


def _merge_native_outputs(existing: object | None, new_output: object) -> object:
    if existing is None:
        return new_output
    if not isinstance(existing, dict) or not isinstance(new_output, dict):
        return new_output

    merged = dict(existing)
    list_keys = (
        "boxes",
        "pred_boxes",
        "out_boxes",
        "out_obj_boxes",
        "scores",
        "confidence",
        "confidences",
        "object_scores",
        "out_scores",
        "out_obj_ids",
        "object_ids",
        "obj_ids",
        "ids",
    )
    mask_keys = (
        "out_binary_masks",
        "binary_masks",
        "masks",
        "pred_masks",
        "out_mask_logits",
        "mask_logits",
    )
    for key in list_keys:
        if key in new_output:
            merged[key] = _as_list(merged.get(key)) + _as_list(new_output.get(key))
    for key in mask_keys:
        if key in new_output:
            merged[key] = _as_mask_list(merged.get(key)) + _as_mask_list(
                new_output.get(key)
            )

    prompt_to_obj_ids = {}
    for source in (existing, new_output):
        raw_mapping = source.get("prompt_to_obj_ids")
        if not isinstance(raw_mapping, dict):
            continue
        for prompt, object_ids in raw_mapping.items():
            prompt_to_obj_ids.setdefault(prompt, [])
            prompt_to_obj_ids[prompt].extend(_as_list(object_ids))
    if prompt_to_obj_ids:
        merged["prompt_to_obj_ids"] = prompt_to_obj_ids
    return merged


def _import_torch() -> object:
    try:
        import torch
    except ImportError as exc:
        raise _dependency_error() from exc
    return torch


def _dependency_error() -> RuntimeError:
    return RuntimeError(
        "SAM3 inference requires PyTorch, Pillow, and a Transformers version with "
        "SAM3 support. Install it with `pip install torch pillow accelerate "
        "\"transformers>=5.0.0\"`, then make sure the VM has Hugging Face access "
        "to `facebook/sam3`."
    )


def _native_dependency_error() -> RuntimeError:
    return RuntimeError(
        "Whole-video SAM3 mode requires Meta's native SAM3 package. Install it "
        "with `pip install 'git+https://github.com/facebookresearch/sam3.git'`, "
        "then make sure the VM has Hugging Face access to the SAM3 checkpoints."
    )


def _read_pil_image(image_path: str | Path) -> object:
    try:
        from PIL import Image
    except ImportError as exc:
        raise _dependency_error() from exc

    with Image.open(image_path) as image:
        return image.convert("RGB").copy()


def _move_inputs_to_model(inputs: object, model: object, default_device: str) -> object:
    if hasattr(inputs, "to"):
        return inputs.to(_model_device(model, default_device))
    return inputs


def _model_device(model: object, default: str) -> object:
    return getattr(model, "device", default)


def _torch_dtype(torch: object, dtype_name: str) -> object | None:
    normalized = dtype_name.strip()
    if not normalized:
        return None
    if normalized in {"auto", "none", "null"}:
        return None
    return getattr(torch, normalized, None)


def _original_sizes(inputs: object) -> object:
    sizes = None
    if hasattr(inputs, "get"):
        sizes = inputs.get("original_sizes")
    elif isinstance(inputs, dict):
        sizes = inputs.get("original_sizes")
    if hasattr(sizes, "detach"):
        sizes = sizes.detach().cpu()
    if hasattr(sizes, "tolist"):
        return sizes.tolist()
    return sizes


def _frame_index(model_outputs: object, fallback: int) -> int:
    raw_value = _result_value(model_outputs, "frame_idx", "frame_index")
    if raw_value is None:
        return fallback
    return int(_to_builtin(raw_value))


def _result_value(result: object, *keys: str) -> object | None:
    if isinstance(result, dict):
        for key in keys:
            if key in result:
                return result[key]
        return None
    for key in keys:
        if hasattr(result, key):
            return getattr(result, key)
    return None


def _as_list(values: object) -> list:
    if values is None:
        return []
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    if hasattr(values, "tolist"):
        converted = values.tolist()
        return converted if isinstance(converted, list) else [converted]
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    if isinstance(values, Iterable) and not isinstance(values, (str, bytes, dict)):
        return list(values)
    return [values]


def _as_mask_list(values: object) -> list:
    if values is None:
        return []
    shape = getattr(values, "shape", None)
    if shape is not None:
        try:
            if len(shape) >= 3:
                return [values[index] for index in range(int(shape[0]))]
        except (TypeError, ValueError):
            pass
    return _as_list(values)


def _parse_box(raw_box: object) -> BoundingBox | None:
    values = _as_list(raw_box)
    if len(values) < 4:
        return None
    x1, y1, x2, y2 = [float(value) for value in values[:4]]
    return BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)


def _bbox_from_mask(mask: object) -> BoundingBox | None:
    if mask is None:
        return None
    try:
        import numpy as np
    except ImportError:
        return _bbox_from_mask_sequence(mask)

    if hasattr(mask, "detach"):
        mask = mask.detach().cpu()
    if hasattr(mask, "numpy"):
        array = mask.numpy()
    else:
        array = np.asarray(mask)

    array = np.squeeze(array)
    if array.ndim != 2:
        return None
    binary_mask = array > 0
    ys, xs = np.where(binary_mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return BoundingBox(
        x1=float(xs.min()),
        y1=float(ys.min()),
        x2=float(xs.max() + 1),
        y2=float(ys.max() + 1),
    )


def _bbox_from_mask_sequence(mask: object) -> BoundingBox | None:
    rows = _squeeze_sequence(mask)
    if not isinstance(rows, list) or not rows:
        return None

    x_values: list[int] = []
    y_values: list[int] = []
    for y, row in enumerate(rows):
        if not isinstance(row, list):
            return None
        for x, value in enumerate(row):
            if value:
                x_values.append(x)
                y_values.append(y)

    if not x_values or not y_values:
        return None
    return BoundingBox(
        x1=float(min(x_values)),
        y1=float(min(y_values)),
        x2=float(max(x_values) + 1),
        y2=float(max(y_values) + 1),
    )


def _squeeze_sequence(value: object) -> object:
    if hasattr(value, "tolist"):
        value = value.tolist()
    while (
        isinstance(value, list)
        and len(value) == 1
        and isinstance(value[0], list)
    ):
        value = value[0]
    return value


def _score_at(scores: list, index: int) -> float:
    if index >= len(scores) or scores[index] is None:
        return 0.0
    return float(_to_builtin(scores[index]))


def _object_id_at(object_ids: list, index: int) -> str:
    if index >= len(object_ids) or object_ids[index] is None:
        return str(index)
    return str(_to_builtin(object_ids[index]))


def _prompt_for_object_id(result: object, object_id: str, default_label: str) -> str:
    prompt_to_obj_ids = _result_value(result, "prompt_to_obj_ids")
    if not isinstance(prompt_to_obj_ids, dict):
        return default_label

    for prompt, raw_ids in prompt_to_obj_ids.items():
        normalized_ids = {str(_to_builtin(raw_id)) for raw_id in _as_list(raw_ids)}
        if object_id in normalized_ids:
            return str(prompt)
    return default_label


def _to_builtin(value: object) -> object:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value
