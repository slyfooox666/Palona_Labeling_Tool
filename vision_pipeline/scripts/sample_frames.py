#!/usr/bin/env python3
"""Placeholder CLI for extracting sample frames from a configured camera."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("camera_config", help="Path to camera YAML config")
    parser.add_argument("--count", type=int, default=10)
    args = parser.parse_args()
    print(
        "Frame sampling is not implemented yet. "
        f"Requested {args.count} frames from {args.camera_config}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
