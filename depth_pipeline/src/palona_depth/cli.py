"""Command-line interface for generating Palona depth feature sidecars."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from palona_depth.control import ControlDataError
from palona_depth.pipeline import BuildOptions, build_depth_features
from palona_depth.runtime_client import RuntimeClientError
from palona_depth.video import VideoDataError


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        payload = build_depth_features(
            BuildOptions(
                video_path=args.video,
                contour_path=args.contour,
                output_path=args.output,
                sample_fps=args.sample_fps,
                start_time=args.start_time,
                end_time=args.end_time,
                max_frames=args.max_frames,
                device=args.device,
                alignment_tolerance_seconds=args.alignment_tolerance,
                person_labels=parse_labels(args.person_labels),
                target_labels=parse_labels(args.target_labels),
                keep_depth_artifacts=args.keep_depth_artifacts,
                stop_runtime=args.stop_runtime,
            ),
            progress=lambda message: print(f"[palona-depth] {message}", file=sys.stderr),
        )
    except (ControlDataError, VideoDataError, RuntimeClientError, ValueError, OSError) as exc:
        print(f"palona-depth: {exc}", file=sys.stderr)
        return 1
    metadata = payload["depth_metadata"]
    print(
        f"Created {args.output} · {len(payload['frames'])} frames · "
        f"{metadata['device']} {metadata['dtype']} · relative depth (not meters)"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="palona-depth",
        description="Build estimated-relative-depth cues from CCTV video and SAM3 Control JSON.",
    )
    parser.add_argument("--video", type=Path, required=True, help="Absolute or relative source video path")
    parser.add_argument("--contour", type=Path, required=True, help="SAM3 Control/Contour JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output depth_features.json sidecar")
    parser.add_argument("--sample-fps", type=float, default=5.0, help="Cue sampling rate; 5 FPS gives 0.2s steps")
    parser.add_argument("--start-time", type=float, default=0.0)
    parser.add_argument("--end-time", type=float)
    parser.add_argument("--max-frames", type=int, help="Bounded smoke-test/debug run")
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--alignment-tolerance", type=float, help="Maximum video/Control timestamp error")
    parser.add_argument("--person-labels", default="person", help="Comma-separated source labels")
    parser.add_argument("--target-labels", default="table", help="Comma-separated target labels")
    parser.add_argument("--keep-depth-artifacts", type=Path, help="Optional private directory for raw NPY/PNG artifacts")
    parser.add_argument("--stop-runtime", action="store_true", help="Release the shared DA3 worker after completion")
    return parser


def parse_labels(value: str) -> tuple[str, ...]:
    labels = tuple(item.strip() for item in value.split(",") if item.strip())
    if not labels:
        raise ValueError("Label lists must not be empty")
    return labels


if __name__ == "__main__":
    raise SystemExit(main())
