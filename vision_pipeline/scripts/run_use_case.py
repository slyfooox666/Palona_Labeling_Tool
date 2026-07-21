#!/usr/bin/env python3
"""Dry-run entrypoint for a configured use case."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from vision_pipeline.core.pipeline import load_pipeline_definition


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to use-case YAML config")
    parser.add_argument("--dry-run", action="store_true", help="Only print resolved summary")
    args = parser.parse_args()

    pipeline = load_pipeline_definition(args.config)
    print(f"use_case_id: {pipeline.use_case_id}")
    print(f"purpose: {pipeline.purpose}")
    print(f"camera_config: {pipeline.config['input']['camera_config']}")
    print(f"detector: {pipeline.config['models']['detector']['config']}")
    print(f"tracker: {pipeline.config['models']['tracker']['config']}")

    vlm = pipeline.config["models"].get("vlm", {})
    if vlm.get("enabled", False):
        print(f"vlm: {vlm['config']}")
        print(f"prompt: {vlm['prompt']}")

    if not args.dry_run:
        raise SystemExit("Runtime execution is not implemented yet. Use --dry-run for now.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
