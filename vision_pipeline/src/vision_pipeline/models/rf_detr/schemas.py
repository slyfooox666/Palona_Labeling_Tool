"""RF-DETR adapter config."""

from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class RFDETRConfig:
    model_id: str
    model_size: str = "large"
    weights_path: str | None = None
    device: str = "cuda"
    endpoint: str | None = None
    input_size: int = 640
    confidence_threshold: float = 0.45
    include_source_image: bool = False
    optimize_for_inference: bool = False
    allowed_classes: tuple[str, ...] = ()
    denied_classes: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, mapping: dict) -> "RFDETRConfig":
        runtime = mapping.get("runtime", {})
        inference = mapping.get("inference", {})
        classes = mapping.get("classes", {})
        endpoint = runtime.get("endpoint")
        endpoint_env = runtime.get("endpoint_env")
        if endpoint is None and endpoint_env:
            endpoint = os.getenv(endpoint_env)
        return cls(
            model_id=mapping.get("model_id", "rf_detr"),
            model_size=str(runtime.get("model_size", "large")),
            weights_path=runtime.get("weights_path") or None,
            device=runtime.get("device", "cuda"),
            endpoint=endpoint,
            input_size=int(inference.get("input_size", 640)),
            confidence_threshold=float(inference.get("confidence_threshold", 0.45)),
            include_source_image=bool(inference.get("include_source_image", False)),
            optimize_for_inference=bool(runtime.get("optimize_for_inference", False)),
            allowed_classes=tuple(classes.get("allow", []) or ()),
            denied_classes=tuple(classes.get("deny", []) or ()),
        )
