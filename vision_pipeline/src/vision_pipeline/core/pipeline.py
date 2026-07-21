"""Pipeline assembly primitives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vision_pipeline.core.config import load_yaml, validate_use_case_config


@dataclass(frozen=True)
class PipelineDefinition:
    """Loaded use-case config with its source path."""

    path: Path
    config: dict[str, Any]

    @property
    def use_case_id(self) -> str:
        return str(self.config["use_case_id"])

    @property
    def purpose(self) -> str:
        return str(self.config["purpose"])


def load_pipeline_definition(path: str | Path) -> PipelineDefinition:
    """Load and validate a use-case config."""
    config_path = Path(path).resolve()
    config = load_yaml(config_path)
    errors = validate_use_case_config(config)
    if errors:
        formatted = "\n".join(f"- {error}" for error in errors)
        raise ValueError(f"Invalid use-case config {config_path}:\n{formatted}")
    return PipelineDefinition(path=config_path, config=config)
