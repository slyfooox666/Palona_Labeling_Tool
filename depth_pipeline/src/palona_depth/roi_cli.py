"""CLI for offline polygon ROI export."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from palona_depth.control import ControlDataError
from palona_depth.roi_export import RoiExportError, RoiExportOptions, export_roi
from palona_depth.video import VideoDataError


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = export_roi(
            RoiExportOptions(
                video_path=args.video,
                contour_path=args.contour,
                roi_path=args.roi,
                masked_video_path=args.masked_video,
                filtered_contour_path=args.filtered_contour,
                codec=args.codec,
            ),
            progress=lambda message: print(f"[palona-roi] {message}", file=sys.stderr),
        )
    except (ControlDataError, VideoDataError, RoiExportError, OSError, ValueError) as exc:
        print(f"palona-roi-export: {exc}", file=sys.stderr)
        return 1

    print(
        f"Created {result['masked_video']} and {result['filtered_contour']} · "
        f"{result['video_frame_count']} video frames · "
        f"{result['kept_track_appearances']} kept / "
        f"{result['removed_track_appearances']} removed track appearances"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="palona-roi-export",
        description=(
            "Black out video pixels outside a normalized polygon ROI and stream-filter "
            "SAM3 Control JSON using the contour-centroid-inside rule."
        ),
    )
    parser.add_argument("--video", type=Path, required=True, help="Source CCTV video (never overwritten)")
    parser.add_argument("--contour", type=Path, required=True, help="SAM3 Control/Contour JSON")
    parser.add_argument(
        "--roi",
        type=Path,
        required=True,
        help="Project JSON with embedded ROI, standalone ROI JSON, or project beside roi.json",
    )
    parser.add_argument("--masked-video", type=Path, required=True, help="Output .mp4/.mov/.mkv video")
    parser.add_argument(
        "--filtered-contour",
        type=Path,
        required=True,
        help="Output normalized tracks Control JSON reloadable by the labeling UI",
    )
    parser.add_argument("--codec", default="libx264", help="PyAV/FFmpeg video encoder (default: libx264)")
    return parser


if __name__ == "__main__":
    raise SystemExit(main())
