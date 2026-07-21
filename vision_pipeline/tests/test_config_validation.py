from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from vision_pipeline.core.config import load_yaml, validate_use_case_config


def test_sample_use_case_configs_are_valid() -> None:
    config_dir = Path(__file__).resolve().parents[1] / "configs" / "use_cases"
    for config_path in config_dir.glob("*.yaml"):
        config = load_yaml(config_path)
        assert validate_use_case_config(config) == []
