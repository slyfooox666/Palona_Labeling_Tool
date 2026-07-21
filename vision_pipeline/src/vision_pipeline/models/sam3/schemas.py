"""SAM3 adapter config."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SAM3Config:
    model_id: str
    model_name: str = "facebook/sam3"
    device: str = "cuda"
    device_map: str | None = "auto"
    dtype: str = "bfloat16"
    processing_device: str = "cpu"
    video_storage_device: str = "cpu"
    inference_state_device: str | None = None
    max_vision_features_cache_size: int = 1
    native_version: str = "sam3"
    checkpoint_path: str | None = None
    compile_model: bool = False
    warm_up: bool = False
    async_loading_frames: bool = True
    max_num_objects: int = 16
    multiplex_count: int = 16
    gpus_to_use: tuple[int, ...] | None = None
    prompts: tuple[str, ...] = ("person",)
    confidence_threshold: float = 0.5
    mask_threshold: float = 0.5
    max_frame_num_to_track: int | None = None
    show_progress_bar: bool = False
    include_masks: bool = False

    @classmethod
    def from_mapping(cls, mapping: dict) -> "SAM3Config":
        runtime = mapping.get("runtime", {})
        inference = mapping.get("inference", {})
        return cls(
            model_id=mapping.get("model_id", "sam3"),
            model_name=str(runtime.get("model_name", "facebook/sam3")),
            device=str(runtime.get("device", "cuda")),
            device_map=_optional_string(runtime.get("device_map", "auto")),
            dtype=str(runtime.get("dtype", "bfloat16")),
            processing_device=str(runtime.get("processing_device", "cpu")),
            video_storage_device=str(runtime.get("video_storage_device", "cpu")),
            inference_state_device=_optional_string(runtime.get("inference_state_device")),
            max_vision_features_cache_size=int(
                runtime.get("max_vision_features_cache_size", 1)
            ),
            native_version=str(
                runtime.get(
                    "native_version",
                    _native_version_from_model_name(
                        str(runtime.get("model_name", "facebook/sam3"))
                    ),
                )
            ),
            checkpoint_path=_optional_string(runtime.get("checkpoint_path")),
            compile_model=bool(runtime.get("compile_model", False)),
            warm_up=bool(runtime.get("warm_up", False)),
            async_loading_frames=bool(runtime.get("async_loading_frames", True)),
            max_num_objects=int(runtime.get("max_num_objects", 16)),
            multiplex_count=int(runtime.get("multiplex_count", 16)),
            gpus_to_use=_optional_int_tuple(runtime.get("gpus_to_use")),
            prompts=_normalize_prompts(inference.get("prompts", ("person",))),
            confidence_threshold=float(inference.get("confidence_threshold", 0.5)),
            mask_threshold=float(inference.get("mask_threshold", 0.5)),
            max_frame_num_to_track=_optional_int(
                inference.get("max_frame_num_to_track")
            ),
            show_progress_bar=bool(inference.get("show_progress_bar", False)),
            include_masks=bool(inference.get("include_masks", False)),
        )


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def _optional_int_tuple(value: object) -> tuple[int, ...] | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    else:
        parts = list(value)  # type: ignore[arg-type]
    values = tuple(int(part) for part in parts if str(part).strip())
    return values or None


def _native_version_from_model_name(model_name: str) -> str:
    lowered = model_name.lower()
    if "sam3.1" in lowered:
        return "sam3.1"
    return "sam3"


def _normalize_prompts(value: object) -> tuple[str, ...]:
    if value is None:
        return ("person",)
    if isinstance(value, str):
        prompts = [value]
    else:
        prompts = list(value)

    normalized = tuple(str(prompt).strip() for prompt in prompts if str(prompt).strip())
    if not normalized:
        raise ValueError("SAM3 config must contain at least one text prompt.")
    return normalized
