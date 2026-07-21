"""Qwen3-VL adapter config."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class Qwen3VLConfig:
    model_id: str
    endpoint: str
    model_name: str
    temperature: float = 0.0
    max_tokens: int = 8192
    timeout_seconds: int = 30
    response_format: str | None = "json_object"
    api_key: str | None = None
    default_target: str = "person"
    bbox_coordinate_format: str = "absolute"
    detection_prompt_template: str = (
        "Find every object matching this target: {target}.\n"
        "Return JSON only with this exact schema: "
        "{\"detections\":[{\"label\":\"string\","
        "\"confidence\":0.0,\"bbox_xyxy\":[x1,y1,x2,y2]}]}.\n"
        "Return at most 20 detections, and do not repeat the same object.\n"
        "Use absolute pixel coordinates in the original image size {width}x{height}. "
        "If there are no matching objects, return {\"detections\":[]}. "
        "Do not include markdown fences or explanatory text."
    )

    @classmethod
    def from_mapping(cls, mapping: dict, endpoint: str | None = None) -> "Qwen3VLConfig":
        runtime = mapping.get("runtime", {})
        request = mapping.get("request", {})
        inference = mapping.get("inference", {})
        resolved_endpoint = _optional_string(endpoint) or _optional_string(
            runtime.get("endpoint")
        )
        endpoint_env = runtime.get("endpoint_env")
        if resolved_endpoint is None and endpoint_env:
            resolved_endpoint = _optional_string(os.getenv(endpoint_env))

        api_key = _optional_string(runtime.get("api_key"))
        api_key_env = runtime.get("api_key_env")
        if api_key is None and api_key_env:
            api_key = _optional_string(os.getenv(api_key_env))

        if not resolved_endpoint:
            raise ValueError(
                "Qwen3-VL endpoint is required. Set runtime.endpoint, pass --endpoint, "
                "or export the configured endpoint_env."
            )
        return cls(
            model_id=mapping.get("model_id", "qwen3_vl"),
            endpoint=resolved_endpoint,
            model_name=runtime.get("model_name", "Qwen/Qwen3-VL-8B-Instruct"),
            temperature=float(request.get("temperature", 0.0)),
            max_tokens=int(request.get("max_tokens", 8192)),
            timeout_seconds=int(request.get("timeout_seconds", 30)),
            response_format=_optional_string(request.get("response_format", "json_object")),
            api_key=api_key,
            default_target=str(inference.get("default_target", "person")),
            bbox_coordinate_format=str(
                inference.get("bbox_coordinate_format", "absolute")
            ),
            detection_prompt_template=str(
                inference.get("detection_prompt_template", cls.detection_prompt_template)
            ),
        )


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text
