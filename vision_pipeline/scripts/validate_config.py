#!/usr/bin/env python3
"""Validate a vision-pipeline use-case config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from vision_pipeline.core.config import load_yaml, validate_use_case_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to use-case YAML config")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_yaml(config_path)
    errors = validate_use_case_config(config)
    if errors:
        print(f"Invalid config: {config_path}")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"OK: {config_path}")
    print(f"use_case_id: {config['use_case_id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
