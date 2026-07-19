"""CLI for safe native SAM3 whole-video preprocessing."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys

from palona_depth.control import ControlDataError
from palona_depth.full_video import FullVideoError, FullVideoOptions, preflight, run_full_video
from palona_depth.video import VideoDataError


def main(argv: list[str] | None = None) -> int:
    os.umask(0o077)
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        root = resolve_pipeline_root(args.vision_pipeline_root)
        python = resolve_pipeline_python(args.pipeline_python, root)
        options = FullVideoOptions(
            video_path=args.video,
            control_path=args.control,
            output_dir=args.output_dir,
            vision_pipeline_root=root,
            pipeline_python=python,
            prompts=tuple(args.prompt),
            required_labels=tuple(args.require_label),
            split_seconds=args.split_seconds,
            overlap_seconds=args.overlap_seconds,
            contour_epsilon_px=args.contour_epsilon_px,
            model_config=args.model_config,
            force=args.force,
        )
        if args.check_only:
            preflight(options, require_cuda=not args.allow_non_cuda)
            print("Full-video preflight passed")
            return 0
        coverage = run_full_video(options, require_cuda=not args.allow_non_cuda)
    except (ControlDataError, VideoDataError, FullVideoError, ValueError, OSError) as exc:
        print(f"palona-full-video: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(coverage), indent=2))
    print(f"Created complete Control JSON: {args.control.expanduser().resolve()}", file=sys.stderr)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="palona-full-video",
        description="Run vision_pipeline SAM3 in native whole-video chunks and validate the Control output.",
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True, help="Final *.control.json path")
    parser.add_argument("--output-dir", type=Path, required=True, help="Private artifacts/log directory outside Git")
    parser.add_argument("--prompt", action="append", required=True, help="Repeat for each SAM3 prompt")
    parser.add_argument(
        "--require-label",
        action="append",
        default=[],
        help="Fail validation when this expected label never appears; repeat as needed",
    )
    parser.add_argument("--split-seconds", type=float, default=12.0)
    parser.add_argument("--overlap-seconds", type=float, default=2.0)
    parser.add_argument("--contour-epsilon-px", type=float, default=2.0)
    parser.add_argument("--vision-pipeline-root", type=Path)
    parser.add_argument("--pipeline-python", type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--check-only", action="store_true", help="Run dependency/CUDA checks without inference")
    parser.add_argument("--force", action="store_true", help="Atomically replace an existing Control JSON")
    parser.add_argument(
        "--allow-non-cuda",
        action="store_true",
        help="Development/testing only: bypass the CUDA preflight (native SAM3 may still require CUDA)",
    )
    return parser


def resolve_pipeline_root(configured: Path | None) -> Path:
    if configured is not None:
        return configured.expanduser().resolve()
    env_value = os.environ.get("VISION_PIPELINE_ROOT")
    if env_value:
        return Path(env_value).expanduser().resolve()
    return Path(__file__).resolve().parents[4] / "vision_pipeline"


def resolve_pipeline_python(configured: Path | None, root: Path) -> Path:
    if configured is not None:
        return configured.expanduser().resolve()
    env_value = os.environ.get("VISION_PIPELINE_PYTHON")
    if env_value:
        return Path(env_value).expanduser().resolve()
    candidates = [
        root / ".venv" / "bin" / "python",
        Path(found) if (found := shutil.which("python3.12")) else None,
        Path(found) if (found := shutil.which("python3")) else None,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.expanduser().is_file():
            return candidate.expanduser().resolve()
    raise FullVideoError("Could not find vision_pipeline Python; set VISION_PIPELINE_PYTHON")


if __name__ == "__main__":
    raise SystemExit(main())
