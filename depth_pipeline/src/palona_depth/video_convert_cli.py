"""CLI for creating a browser-compatible H.264 MP4 playback copy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from palona_depth.video_convert import VideoConvertError, VideoConvertOptions, convert_video


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = convert_video(
            VideoConvertOptions(
                input_path=args.input,
                output_path=args.output,
                codec=args.codec,
                crf=args.crf,
                preset=args.preset,
            ),
            progress=lambda message: print(f"[palona-video] {message}", file=sys.stderr),
        )
    except (VideoConvertError, OSError, ValueError) as exc:
        print(f"palona-video-convert: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="palona-video-convert",
        description=(
            "Create an atomic H.264/yuv420p MP4 playback copy while preserving "
            "frame order, FPS, dimensions, and duration. Audio is not preserved."
        ),
    )
    parser.add_argument("--input", type=Path, required=True, help="Source video, such as scene.mkv")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Same-stem .mp4 output, such as scene.mp4",
    )
    parser.add_argument("--codec", default="libx264", help="PyAV encoder (default: libx264)")
    parser.add_argument("--crf", type=int, default=18, help="H.264 CRF, 0-51 (default: 18)")
    parser.add_argument("--preset", default="medium", help="libx264 preset (default: medium)")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
