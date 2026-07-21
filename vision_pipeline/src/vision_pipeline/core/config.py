"""Config loading and lightweight validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file as a mapping."""
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required for YAML configs. Install it with `pip install pyyaml`."
        ) from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at top level of {config_path}")
    return data


def resolve_config_path(base_file: str | Path, maybe_relative_path: str | Path) -> Path:
    """Resolve a referenced config path relative to the file that referenced it."""
    ref = Path(maybe_relative_path)
    if ref.is_absolute():
        return ref
    return Path(base_file).resolve().parent.joinpath(ref).resolve()


def validate_use_case_config(config: dict[str, Any]) -> list[str]:
    """Return validation errors for a use-case config."""
    errors: list[str] = []

    for key in ("config_version", "use_case_id", "purpose", "input", "models"):
        if key not in config:
            errors.append(f"missing top-level key: {key}")

    input_config = config.get("input", {})
    if not isinstance(input_config, dict):
        errors.append("input must be a mapping")
    elif "camera_config" not in input_config:
        errors.append("input.camera_config is required")

    models = config.get("models", {})
    if not isinstance(models, dict):
        errors.append("models must be a mapping")
        return errors

    has_detector_tracker = "detector" in models or "tracker" in models
    has_sam3 = "sam3" in models

    if not has_detector_tracker and not has_sam3:
        errors.append("models must define detector/tracker or sam3")

    if has_detector_tracker:
        for model_key in ("detector", "tracker"):
            model_config = models.get(model_key, {})
            if not isinstance(model_config, dict):
                errors.append(f"models.{model_key} must be a mapping")
            elif "config" not in model_config:
                errors.append(f"models.{model_key}.config is required")

    if has_sam3:
        sam3_config = models.get("sam3", {})
        if not isinstance(sam3_config, dict):
            errors.append("models.sam3 must be a mapping")
        elif "config" not in sam3_config:
            errors.append("models.sam3.config is required")

    vlm_config = models.get("vlm")
    if isinstance(vlm_config, dict) and vlm_config.get("enabled", False):
        if "config" not in vlm_config:
            errors.append("models.vlm.config is required when VLM is enabled")
        if "prompt" not in vlm_config:
            errors.append("models.vlm.prompt is required when VLM is enabled")

    return errors
