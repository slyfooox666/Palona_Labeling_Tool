#!/usr/bin/env python3
"""Vision pipeline command-line entry point."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from bisect import bisect_left
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from vision_pipeline.core.config import load_yaml, resolve_config_path
from vision_pipeline.core.roi import (
    ROI,
    crop_frames_to_roi,
    crop_image_to_roi,
    offset_bbox,
    offset_detections,
    offset_tracks,
    parse_roi,
)
from vision_pipeline.core.sam_live import (
    FrameWindow,
    Sam3LiveConfig,
    Sam3TrackStitcher,
    StitchWindowResult,
    build_frame_windows,
)
from vision_pipeline.core.schemas import Detection, Track
from vision_pipeline.models.bot_sort import BoTSORTTracker
from vision_pipeline.models.bot_sort.schemas import BoTSORTConfig
from vision_pipeline.models.qwen3_vl import Qwen3VLClient, Qwen3VLConfig
from vision_pipeline.models.rf_detr import RFDETRDetector
from vision_pipeline.models.rf_detr.schemas import RFDETRConfig
from vision_pipeline.models.sam3 import SAM3Config, SAM3ModelAdapter
from vision_pipeline.models.sam3.adapter import Sam3FrameResult
from vision_pipeline.utils.video import (
    SampledFrame,
    extract_frames_at_fps,
    extract_sample_frames,
    probe_video,
)

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
SAM3_SPLIT_TRACK_MATCH_IOU_THRESHOLD = 0.2
ANNOTATION_VIDEO_CRF = 23
ANNOTATION_VIDEO_PRESET = "veryfast"


def main() -> int:
    parser = argparse.ArgumentParser(prog="vp_cli.py")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_rfdetr_parser(subparsers)
    add_sam3_parser(subparsers)
    add_sam3_live_parser(subparsers)
    add_qwen3_vl_parser(subparsers)

    args = parser.parse_args()
    if args.command == "rfdetr":
        return run_rfdetr_command(args)
    if args.command == "sam3":
        return run_sam3_command(args)
    if args.command == "sam3-live":
        return run_sam3_live_command(args)
    if args.command in {"qwen3-vl", "qwen3"}:
        return run_qwen3_vl_command(args)
    raise AssertionError(f"Unhandled command: {args.command}")


def add_rfdetr_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "rfdetr",
        help="Run RF-DETR detection and optional BoT-SORT tracking.",
    )
    parser.add_argument("input", help="Path to an image or video")
    parser.add_argument(
        "--model-config",
        default="vision_pipeline/configs/models/rf_detr.yaml",
        help="Path to RF-DETR model YAML",
    )
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=None,
        help="For video inputs, sample frames at this FPS instead of using --sample-count.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to sample when using --sample-fps.",
    )
    add_frame_extraction_arguments(parser)
    parser.add_argument("--classes", nargs="*", default=None)
    parser.add_argument(
        "--tracker",
        choices=("none", "bot-sort"),
        default="none",
        help="Optional tracker to run after RF-DETR detections.",
    )
    parser.add_argument(
        "--tracker-config",
        default="vision_pipeline/configs/models/bot_sort.yaml",
        help="Path to BoT-SORT tracker YAML.",
    )
    parser.add_argument(
        "--track-classes",
        nargs="*",
        default=["person"],
        help="Detection labels to feed into the tracker.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--draw-boxes",
        action="store_true",
        help="Write annotated images with detection bounding boxes and confidence labels.",
    )
    parser.add_argument(
        "--tracked-object-clips",
        action="store_true",
        help=(
            "Write one MP4 per tracked object continuous appearance. "
            "Requires --tracker bot-sort for RF-DETR video inputs."
        ),
    )
    parser.add_argument(
        "--single-appearance",
        action="store_true",
        help=(
            "With --tracked-object-clips, write one MP4 per track ID spanning "
            "first visible frame through final tracked frame, including hidden frames."
        ),
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only prepare/print input frames; do not load RF-DETR.",
    )
    add_timeit_argument(parser)


def add_sam3_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "sam3",
        help="Run SAM3 promptable image segmentation or video tracking.",
    )
    parser.add_argument("input", help="Path to an image or video")
    parser.add_argument(
        "--model-config",
        default="vision_pipeline/configs/models/sam3.yaml",
        help="Path to SAM3 model YAML",
    )
    parser.add_argument(
        "--prompt",
        nargs="+",
        default=None,
        help=(
            "Text prompt(s) to segment/track, for example: --prompt person "
            "or --prompt \"bamboo steamer\""
        ),
    )
    parser.add_argument(
        "--roi",
        default=None,
        help=(
            "Optional full-frame ROI for SAM3 input as x1,y1,x2,y2. "
            "SAM3 runs on the crop, then boxes are mapped back to original coordinates."
        ),
    )
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument(
        "--video-mode",
        choices=("sampled", "whole"),
        default="sampled",
        help=(
            "For video inputs, sampled extracts frames first; whole sends the "
            "video file directly to SAM3's native video session."
        ),
    )
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=None,
        help="For video inputs, sample frames at this FPS instead of using --sample-count.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help=(
            "Maximum frames to sample when using --sample-fps, or maximum "
            "whole-video frames to process."
        ),
    )
    parser.add_argument(
        "--split-seconds",
        type=positive_float,
        default=None,
        help=(
            "With --video-mode whole, split the input into chunks of at most "
            "this many seconds before running SAM3."
        ),
    )
    parser.add_argument(
        "--overlap-seconds",
        type=nonnegative_float,
        default=0.0,
        help=(
            "With --split-seconds, overlap consecutive chunks by this many "
            "seconds and use the overlap to keep track IDs consistent."
        ),
    )
    add_frame_extraction_arguments(parser)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--draw-boxes",
        action="store_true",
        help="Write annotated images with SAM3 bounding boxes and confidence labels.",
    )
    parser.add_argument(
        "--draw-contours",
        action="store_true",
        help=(
            "Draw SAM3 mask contours when masks are available. In --video-mode "
            "whole, this writes an annotated MP4."
        ),
    )
    parser.add_argument(
        "--contour-json",
        nargs="?",
        const="true",
        default=None,
        metavar="PATH|true",
        help=(
            "Include simplified mask contour coordinates per detection/track. "
            "Use --contour-json or --contour-json true to print JSON to stdout, "
            "or pass a path to write the JSON there."
        ),
    )
    parser.add_argument(
        "--contour-epsilon-px",
        type=float,
        default=2.0,
        help=(
            "Douglas-Peucker contour simplification epsilon in pixels for "
            "--contour-json (default: 2.0; use 0 for unsimplified contours)."
        ),
    )
    parser.add_argument(
        "--annotated-video",
        action="store_true",
        help=(
            "With --video-mode whole, write an annotated MP4. "
            "--draw-contours implies this."
        ),
    )
    parser.add_argument(
        "--tracked-object-clips",
        action="store_true",
        help="Write one MP4 per tracked object continuous appearance.",
    )
    parser.add_argument(
        "--single-appearance",
        action="store_true",
        help=(
            "With --tracked-object-clips, write one MP4 per track ID spanning "
            "first visible frame through final tracked frame, including hidden frames."
        ),
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only prepare/print input frames; do not load SAM3.",
    )
    add_timeit_argument(parser)


def add_sam3_live_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "sam3-live",
        help="Run SAM3 windowed tracking with global dwell-time stitching.",
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Video file path or RTSP URL. Overrides the source from --config.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Use-case YAML containing sam3_live settings.",
    )
    parser.add_argument(
        "--model-config",
        default=None,
        help="Path to SAM3 model YAML. Defaults to config models.sam3.config.",
    )
    parser.add_argument(
        "--prompt",
        nargs="+",
        default=None,
        help="SAM prompt(s), for example: --prompt basket.",
    )
    parser.add_argument(
        "--roi",
        default=None,
        help="Optional full-frame ROI as x1,y1,x2,y2. Overrides config sam3_live.roi.",
    )
    parser.add_argument(
        "--strategy",
        choices=("long_window", "rolling_window", "per_frame"),
        default=None,
        help="SAM processing strategy. Defaults to config sam3_live.strategy.",
    )
    parser.add_argument("--sample-fps", type=float, default=None)
    parser.add_argument("--window-seconds", type=float, default=None)
    parser.add_argument("--stride-seconds", type=float, default=None)
    parser.add_argument("--dwell-threshold-seconds", type=float, default=None)
    parser.add_argument("--missing-grace-seconds", type=float, default=None)
    parser.add_argument("--match-iou-threshold", type=float, default=None)
    parser.add_argument("--match-distance-threshold-px", type=float, default=None)
    parser.add_argument(
        "--max-windows",
        type=int,
        default=None,
        help="Stop after this many SAM windows. Useful for RTSP smoke tests.",
    )
    parser.add_argument(
        "--max-runtime-seconds",
        type=float,
        default=None,
        help="For file inputs, sample only this many seconds.",
    )
    add_frame_extraction_arguments(parser)
    parser.add_argument(
        "--rtsp-smoke-test",
        action="store_true",
        help=(
            "For RTSP inputs, only open the stream, decode a few frames, "
            "write optional ROI crops, and exit without SAM/window processing."
        ),
    )
    parser.add_argument(
        "--smoke-test-frames",
        type=int,
        default=3,
        help="Number of frames to decode for --rtsp-smoke-test.",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--draw-boxes",
        action="store_true",
        help="Write annotated ROI frames with global track IDs.",
    )
    parser.add_argument(
        "--evidence-video",
        action="store_true",
        help="Create an MP4 from annotated committed ROI frames.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only prepare/print windows; do not load SAM3.",
    )
    add_timeit_argument(parser)


def add_qwen3_vl_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "qwen3-vl",
        aliases=["qwen3"],
        help="Run Qwen3-VL prompt-grounded image/frame analysis.",
    )
    parser.add_argument("input", help="Path to an image or video")
    parser.add_argument(
        "--model-config",
        default="vision_pipeline/configs/models/qwen3_vl.yaml",
        help="Path to Qwen3-VL model YAML",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Override Qwen3-VL endpoint, otherwise use QWEN3_VL_ENDPOINT.",
    )
    parser.add_argument(
        "--task",
        choices=("detect", "ask"),
        default="detect",
        help="detect returns bounding boxes; ask returns raw model JSON per image/frame.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help=(
            "Detection target for --task detect, or full instruction for --task ask. "
            "Quote multi-word prompts."
        ),
    )
    parser.add_argument(
        "--bbox-format",
        choices=("absolute", "normalized", "qwen1000", "auto"),
        default=None,
        help="Coordinate format to use when parsing Qwen3-VL detection boxes.",
    )
    parser.add_argument("--sample-count", type=int, default=5)
    parser.add_argument(
        "--sample-fps",
        type=float,
        default=None,
        help="For video inputs, sample frames at this FPS instead of using --sample-count.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to sample when using --sample-fps.",
    )
    add_frame_extraction_arguments(parser)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--draw-boxes",
        action="store_true",
        help="For --task detect, write annotated images with Qwen3-VL boxes.",
    )
    parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only prepare/print input frames; do not call Qwen3-VL.",
    )
    add_timeit_argument(parser)


def add_timeit_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--timeit",
        action="store_true",
        help="Print average per-frame inference/API latency at the end.",
    )


def add_frame_extraction_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--number-worker-threads",
        type=positive_int,
        default=8,
        help="Number of worker threads to use for frame extraction. Defaults to 8.",
    )
    parser.add_argument(
        "--start-from",
        type=nonnegative_float,
        default=0.0,
        help="Start processing video inputs from this source-video offset in seconds.",
    )


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def contour_json_requested(args: argparse.Namespace) -> bool:
    value = getattr(args, "contour_json", None)
    if value is None:
        return False
    return str(value).strip().lower() not in {"false", "0", "no", "off"}


def contour_json_output_path(args: argparse.Namespace) -> Path | None:
    value = getattr(args, "contour_json", None)
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in {"", "true", "1", "yes", "on", "-", "stdout", "terminal"}:
        return None
    if text.lower() in {"false", "0", "no", "off"}:
        return None
    return Path(text)


def emit_json_result(result: dict, args: argparse.Namespace | None = None) -> None:
    output_path = contour_json_output_path(args) if args is not None else None
    if output_path is None:
        json.dump(result, sys.stdout, indent=2)
        print()
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(result, output_file, indent=2)
        output_file.write("\n")
    print(f"json_output_path: {output_path}")


def run_rfdetr_command(args: argparse.Namespace) -> int:

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    input_type = detect_input_type(input_path)

    if input_type == "image":
        return run_image(input_path, args)
    if input_type == "video":
        return run_video(input_path, args)
    raise AssertionError(f"Unhandled input type: {input_type}")


def run_sam3_command(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    input_type = detect_input_type(input_path)

    if input_type == "image":
        return run_sam3_image(input_path, args)
    if input_type == "video":
        return run_sam3_video(input_path, args)
    raise AssertionError(f"Unhandled input type: {input_type}")


def run_sam3_live_command(args: argparse.Namespace) -> int:
    use_case_config_path = Path(args.config) if args.config else None
    use_case_mapping = load_yaml(use_case_config_path) if use_case_config_path else {}
    live_config = build_sam3_live_config(args, use_case_mapping)
    source_uri, source_type = resolve_sam3_live_source(
        args,
        use_case_mapping,
        use_case_config_path,
    )
    model_config_path = resolve_sam3_live_model_config(
        args,
        use_case_mapping,
        use_case_config_path,
    )

    with tempfile.TemporaryDirectory(prefix="vision_pipeline_sam3_live_") as temp_dir:
        output_dir = Path(args.output_dir) if args.output_dir else Path(temp_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"sam3-live: source_type={source_type} strategy={live_config.strategy} "
            f"sample_fps={live_config.sample_fps:.2f}"
        )
        if live_config.roi:
            print(f"roi_xyxy: {live_config.roi}")

        if args.rtsp_smoke_test:
            if source_type != "rtsp":
                raise ValueError("--rtsp-smoke-test requires an RTSP input source.")
            result = run_sam3_live_rtsp_smoke_test(
                source_uri,
                live_config,
                output_dir,
                args,
            )
        elif source_type == "rtsp":
            result = run_sam3_live_rtsp(
                source_uri,
                live_config,
                model_config_path,
                output_dir,
                args,
            )
        else:
            result = run_sam3_live_file(
                Path(source_uri),
                live_config,
                model_config_path,
                output_dir,
                args,
            )

        print(json.dumps(result, indent=2))
    return 0


def run_qwen3_vl_command(args: argparse.Namespace) -> int:
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    input_type = detect_input_type(input_path)

    if input_type == "image":
        return run_qwen3_vl_image(input_path, args)
    if input_type == "video":
        return run_qwen3_vl_video(input_path, args)
    raise AssertionError(f"Unhandled input type: {input_type}")


def run_image(image_path: Path, args: argparse.Namespace) -> int:
    print(f"image: {image_path}")

    if args.extract_only:
        print(f"0.00s {image_path}")
        return 0

    timer = LatencyTimer(enabled=args.timeit)
    detector = build_detector(args.model_config, detection_classes(args))
    tracker = build_tracker(args)
    with timer.measure():
        detections = detector.detect_image_path(image_path, timestamp_seconds=0.0)
        tracks = (
            tracker.update_with_image(
                trackable_detections(detections, args),
                timestamp_seconds=0.0,
                image_path=image_path,
            )
            if tracker
            else []
        )
    annotated_image_path = None
    if args.draw_boxes:
        annotated_image_path = output_image_path(image_path, args.output_dir)
        if tracks:
            draw_tracks(image_path, annotated_image_path, tracks)
        else:
            draw_detections(image_path, annotated_image_path, detections)

    print(
        json.dumps(
            {
                "input_type": "image",
                "image_path": str(image_path),
                "annotated_image_path": (
                    str(annotated_image_path) if annotated_image_path else None
                ),
                "detections": serialize_detections(detections),
                "tracks": serialize_tracks(tracks),
            },
            indent=2,
        )
    )
    timer.print_summary()
    return 0


def run_video(video_path: Path, args: argparse.Namespace) -> int:
    if args.tracked_object_clips and args.tracker == "none":
        raise ValueError("--tracked-object-clips requires --tracker bot-sort.")
    if args.single_appearance and not args.tracked_object_clips:
        raise ValueError("--single-appearance requires --tracked-object-clips.")

    metadata = probe_video(video_path)
    print(
        f"video: {metadata.path} "
        f"{metadata.width}x{metadata.height} "
        f"{metadata.duration_seconds:.2f}s "
        f"{metadata.fps:.2f}fps"
    )

    with tempfile.TemporaryDirectory(prefix="vision_pipeline_rfdetr_") as temp_dir:
        output_dir = resolve_video_output_dir(
            video_path,
            args,
            temp_dir,
            persistent_default=args.tracked_object_clips,
        )
        if args.sample_fps is not None:
            frames = extract_frames_at_fps(
                video_path,
                output_dir,
                sample_fps=args.sample_fps,
                max_frames=args.max_frames,
                start_from_seconds=args.start_from,
                worker_threads=args.number_worker_threads,
            )
        else:
            frames = extract_sample_frames(
                video_path,
                output_dir,
                args.sample_count,
                start_from_seconds=args.start_from,
                worker_threads=args.number_worker_threads,
            )
        print(f"sampled_frames: {len(frames)}")

        if args.extract_only:
            for frame in frames:
                print(f"{frame.timestamp_seconds:.2f}s {frame.path}")
            return 0

        detector = build_detector(args.model_config, detection_classes(args))
        tracker = build_tracker(args)
        timer = LatencyTimer(enabled=args.timeit)

        result = {
            "input_type": "video",
            "video": str(metadata.path),
            "tracker": args.tracker,
            "start_from_seconds": args.start_from,
            "frames": [],
            "tracked_object_clips": [],
        }
        clip_frame_records: list[TrackedFrameRecord] = []
        for frame in frames:
            with timer.measure():
                detections = detector.detect_image_path(
                    frame.path,
                    frame.timestamp_seconds,
                )
                tracks = (
                    tracker.update_with_image(
                        trackable_detections(detections, args),
                        timestamp_seconds=frame.timestamp_seconds,
                        image_path=frame.path,
                    )
                    if tracker
                    else []
                )
            annotated_image_path = None
            if args.draw_boxes:
                annotated_image_path = frame.path.with_name(
                    f"{frame.path.stem}_boxes{frame.path.suffix}"
                )
                if tracks:
                    draw_tracks(frame.path, annotated_image_path, tracks)
                else:
                    draw_detections(frame.path, annotated_image_path, detections)

            result["frames"].append(
                {
                    "timestamp_seconds": frame.timestamp_seconds,
                    "image_path": str(frame.path),
                    "annotated_image_path": (
                        str(annotated_image_path) if annotated_image_path else None
                    ),
                    "detections": serialize_detections(detections),
                    "tracks": serialize_tracks(tracks),
                }
            )
            if args.tracked_object_clips:
                clip_frame_records.append(
                    TrackedFrameRecord(
                        image_path=frame.path,
                        timestamp_seconds=frame.timestamp_seconds,
                        tracks=tracks,
                    )
                )

        if args.tracked_object_clips:
            result["tracked_object_clips"] = write_tracked_object_clips(
                source_video_path=video_path,
                frame_records=clip_frame_records,
                output_dir=output_dir,
                output_fps=infer_frame_sequence_fps(frames, fallback=args.sample_fps),
                single_appearance=args.single_appearance,
            )

        print(json.dumps(result, indent=2))
        timer.print_summary()
    return 0


def run_sam3_image(image_path: Path, args: argparse.Namespace) -> int:
    print(f"image: {image_path}")

    if args.extract_only:
        print(f"0.00s {image_path}")
        return 0

    roi = parse_roi(args.roi)
    with tempfile.TemporaryDirectory(prefix="vision_pipeline_sam3_roi_") as temp_dir:
        output_dir = Path(args.output_dir) if args.output_dir else Path(temp_dir)
        sam3_image_path = (
            crop_image_to_roi(image_path, output_dir, roi) if roi else image_path
        )
        if roi:
            print(f"roi_xyxy: {roi}")

        timer = LatencyTimer(enabled=args.timeit)
        adapter = build_sam3_adapter(
            args.model_config,
            include_masks=args.draw_contours or contour_json_requested(args),
        )
        with timer.measure():
            crop_detections = adapter.detect_image_path(
                sam3_image_path,
                prompt=args.prompt,
                timestamp_seconds=0.0,
            )
        detections = (
            offset_detections(crop_detections, roi) if roi else crop_detections
        )
        annotated_image_path = None
        annotated_crop_image_path = None
        if args.draw_boxes or args.draw_contours:
            if roi:
                annotated_crop_image_path = annotated_variant_path(sam3_image_path)
                draw_detections(
                    sam3_image_path,
                    annotated_crop_image_path,
                    crop_detections,
                    draw_contours=args.draw_contours,
                )
            annotated_image_path = output_image_path(image_path, args.output_dir)
            draw_detections(
                image_path,
                annotated_image_path,
                detections,
                draw_contours=args.draw_contours,
                mask_offset=(roi[0], roi[1]) if roi else None,
                mask_canvas_size=((roi[2] - roi[0], roi[3] - roi[1]) if roi else None),
            )

        result = {
            "input_type": "image",
            "model": "sam3",
            "image_path": str(image_path),
            "crop_image_path": (
                str(sam3_image_path) if roi and args.output_dir else None
            ),
            "roi_xyxy": roi,
            "annotated_image_path": (
                str(annotated_image_path) if annotated_image_path else None
            ),
            "annotated_crop_image_path": (
                str(annotated_crop_image_path) if annotated_crop_image_path else None
            ),
            "detections": serialize_detections(
                detections,
                include_contours=contour_json_requested(args),
                contour_epsilon_px=args.contour_epsilon_px,
            ),
        }
        emit_json_result(result, args)
        timer.print_summary()
    return 0


def run_sam3_video(video_path: Path, args: argparse.Namespace) -> int:
    if args.single_appearance and not args.tracked_object_clips:
        raise ValueError("--single-appearance requires --tracked-object-clips.")
    if args.split_seconds is not None and args.video_mode != "whole":
        raise ValueError("--split-seconds requires --video-mode whole.")
    if args.overlap_seconds and args.split_seconds is None:
        raise ValueError("--overlap-seconds requires --split-seconds.")
    if args.video_mode == "whole":
        return run_sam3_video_whole(video_path, args)

    metadata = probe_video(video_path)
    print(
        f"video: {metadata.path} "
        f"{metadata.width}x{metadata.height} "
        f"{metadata.duration_seconds:.2f}s "
        f"{metadata.fps:.2f}fps"
    )

    with tempfile.TemporaryDirectory(prefix="vision_pipeline_sam3_") as temp_dir:
        output_dir = resolve_video_output_dir(
            video_path,
            args,
            temp_dir,
            persistent_default=args.tracked_object_clips,
        )
        if args.sample_fps is not None:
            frames = extract_frames_at_fps(
                video_path,
                output_dir,
                sample_fps=args.sample_fps,
                max_frames=args.max_frames,
                start_from_seconds=args.start_from,
                worker_threads=args.number_worker_threads,
            )
        else:
            frames = extract_sample_frames(
                video_path,
                output_dir,
                args.sample_count,
                start_from_seconds=args.start_from,
                worker_threads=args.number_worker_threads,
            )
        print(f"sampled_frames: {len(frames)}")

        roi = parse_roi(args.roi)
        sam3_frames = crop_frames_to_roi(frames, output_dir, roi) if roi else frames
        if roi:
            print(f"roi_xyxy: {roi}")

        if args.extract_only:
            for index, frame in enumerate(frames):
                if roi:
                    print(
                        f"{frame.timestamp_seconds:.2f}s "
                        f"{sam3_frames[index].path} roi_source={frame.path}"
                    )
                else:
                    print(f"{frame.timestamp_seconds:.2f}s {frame.path}")
            return 0

        adapter = build_sam3_adapter(
            args.model_config,
            include_masks=(
                args.draw_contours
                or contour_json_requested(args)
                or args.video_mode == "whole"
            ),
        )
        timer = LatencyTimer(enabled=args.timeit)
        with timer.measure(frame_count=len(sam3_frames)):
            frame_results = adapter.track_video_frames(sam3_frames, prompt=args.prompt)

        result = {
            "input_type": "video",
            "video": str(metadata.path),
            "model": "sam3",
            "roi_xyxy": roi,
            "start_from_seconds": args.start_from,
            "frames": [],
            "tracked_object_clips": [],
        }
        clip_frame_records: list[TrackedFrameRecord] = []
        for index, frame_result in enumerate(frame_results):
            source_frame = frames[index]
            tracks = offset_tracks(frame_result.tracks, roi) if roi else frame_result.tracks
            annotated_image_path = None
            annotated_crop_image_path = None
            if args.draw_boxes or args.draw_contours:
                if roi:
                    annotated_crop_image_path = annotated_variant_path(
                        frame_result.image_path
                    )
                    draw_tracks(
                        frame_result.image_path,
                        annotated_crop_image_path,
                        frame_result.tracks,
                        draw_contours=args.draw_contours,
                    )
                annotated_image_path = source_frame.path.with_name(
                    f"{source_frame.path.stem}_boxes{source_frame.path.suffix}"
                )
                draw_tracks(
                    source_frame.path,
                    annotated_image_path,
                    tracks,
                    draw_contours=args.draw_contours,
                    mask_offset=(roi[0], roi[1]) if roi else None,
                    mask_canvas_size=(
                        (roi[2] - roi[0], roi[3] - roi[1]) if roi else None
                    ),
                )

            result["frames"].append(
                {
                    "timestamp_seconds": frame_result.timestamp_seconds,
                    "image_path": str(source_frame.path),
                    "crop_image_path": str(frame_result.image_path) if roi else None,
                    "annotated_image_path": (
                        str(annotated_image_path) if annotated_image_path else None
                    ),
                    "annotated_crop_image_path": (
                        str(annotated_crop_image_path)
                        if annotated_crop_image_path
                        else None
                    ),
                    "tracks": serialize_tracks(
                        tracks,
                        include_contours=contour_json_requested(args),
                        contour_epsilon_px=args.contour_epsilon_px,
                    ),
                }
            )
            if args.tracked_object_clips:
                clip_frame_records.append(
                    sam3_tracked_clip_frame_record(
                        source_frame,
                        frame_result,
                        tracks,
                        roi,
                    )
                )

        if args.tracked_object_clips:
            result["tracked_object_clips"] = write_tracked_object_clips(
                source_video_path=video_path,
                frame_records=clip_frame_records,
                output_dir=output_dir,
                output_fps=infer_frame_sequence_fps(frames, fallback=args.sample_fps),
                single_appearance=args.single_appearance,
            )

        emit_json_result(result, args)
        timer.print_summary()
    return 0


def run_sam3_video_whole(video_path: Path, args: argparse.Namespace) -> int:
    if args.tracked_object_clips:
        raise ValueError("--tracked-object-clips is only supported in sampled mode.")
    if args.sample_fps is not None and args.sample_fps <= 0:
        raise ValueError("--sample-fps must be positive when set.")
    if args.split_seconds is None:
        if args.sample_fps is not None or args.sample_count != 5:
            print(
                "warning: --video-mode whole without --split-seconds ignores "
                "--sample-fps and --sample-count",
                file=sys.stderr,
            )
    elif args.sample_count != 5:
        print(
            "warning: --video-mode whole with --split-seconds ignores --sample-count",
            file=sys.stderr,
        )
    if args.start_from:
        raise ValueError(
            "--video-mode whole currently processes from the start of the video."
        )

    metadata = probe_video(video_path)
    print(
        f"video: {metadata.path} "
        f"{metadata.width}x{metadata.height} "
        f"{metadata.duration_seconds:.2f}s "
        f"{metadata.fps:.2f}fps"
    )

    with tempfile.TemporaryDirectory(prefix="vision_pipeline_sam3_whole_") as temp_dir:
        output_dir = resolve_video_output_dir(
            video_path,
            args,
            temp_dir,
            persistent_default=(
                not args.extract_only
                and (args.draw_boxes or args.draw_contours or args.annotated_video)
            ),
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        roi = parse_roi(args.roi)
        processing_fps = resolve_sam3_chunk_processing_fps(args, metadata.fps)
        draw_source_video_path = video_path
        sam3_video_path = video_path
        processing_video_path = None
        if processing_fps is not None and not args.extract_only:
            processing_video_path = prepare_sam3_processing_video(
                video_path,
                Path(temp_dir) / "sam3_processing",
                fps=processing_fps,
                max_frames=args.max_frames,
            )
            draw_source_video_path = processing_video_path
            sam3_video_path = processing_video_path
            print(
                f"sam3_processing_video: {processing_video_path} "
                f"fps={processing_fps:g}",
                flush=True,
            )
        if roi:
            print(f"roi_xyxy: {roi}")
            sam3_video_path = crop_video_to_roi(
                processing_video_path or video_path,
                output_dir,
                roi,
            )

        source_frame_timestamps = None
        if processing_fps is None:
            source_frame_timestamps = probe_video_frame_timestamps(sam3_video_path)

        if args.extract_only:
            if args.split_seconds is None:
                print(f"whole_video: {sam3_video_path}")
            else:
                chunks = build_sam3_video_chunk_specs(
                    sam3_video_path,
                    Path(temp_dir) / "sam3_chunks",
                    duration_seconds=metadata.duration_seconds,
                    fps=metadata.fps,
                    split_seconds=args.split_seconds,
                    overlap_seconds=args.overlap_seconds,
                    max_frames=args.max_frames,
                    processing_fps=processing_fps,
                    source_frame_timestamps=source_frame_timestamps,
                )
                for chunk in chunks:
                    print(
                        f"chunk_{chunk.index}: "
                        f"start={chunk.start_seconds:.2f}s "
                        f"duration={chunk.duration_seconds:.2f}s "
                        f"processing_fps={chunk_processing_fps(chunk, metadata.fps):.2f} "
                        f"emit_start_frame={chunk.emit_start_frame_index}"
                    )
            return 0

        adapter = build_sam3_adapter(
            args.model_config,
            include_masks=True,
        )
        timer = LatencyTimer(enabled=args.timeit)
        should_write_video = (
            args.annotated_video or args.draw_contours or args.draw_boxes
        )
        draw_contours_for_video = should_write_video and (
            args.draw_contours or not args.draw_boxes
        )
        need_contours = contour_json_requested(args) or draw_contours_for_video
        chunks = None
        if args.split_seconds is not None:
            chunks = build_sam3_video_chunk_specs(
                sam3_video_path,
                Path(temp_dir) / "sam3_chunks",
                duration_seconds=metadata.duration_seconds,
                fps=metadata.fps,
                split_seconds=args.split_seconds,
                overlap_seconds=args.overlap_seconds,
                max_frames=args.max_frames,
                processing_fps=processing_fps,
                source_frame_timestamps=source_frame_timestamps,
            )
            print(
                f"sam3_split: chunks={len(chunks)} "
                f"split_seconds={args.split_seconds:g} "
                f"overlap_seconds={args.overlap_seconds:g}"
            )
            if processing_fps is not None:
                print(
                    f"sam3_sample_fps: {processing_fps:g} "
                    f"source_fps={metadata.fps:.2f}"
                )
            split_sam3_video_chunks(
                sam3_video_path,
                chunks,
                input_fps=(
                    processing_fps
                    if processing_video_path is not None
                    else metadata.fps
                ),
            )
            frame_count_estimate = max(
                sum(chunk.max_frames or 0 for chunk in chunks),
                1,
            )
            with timer.measure(frame_count=frame_count_estimate):
                frame_results = track_and_merge_sam3_video_chunks(
                    adapter,
                    chunks,
                    prompt=args.prompt,
                    fps=metadata.fps,
                    materialize_contours=need_contours,
                    contour_epsilon_px=args.contour_epsilon_px,
                )
                clear_cuda_cache()
        else:
            frame_count_estimate = max(
                int(round(metadata.duration_seconds * metadata.fps)),
                1,
            )
            if args.max_frames is not None:
                frame_count_estimate = min(frame_count_estimate, args.max_frames)
            with timer.measure(frame_count=frame_count_estimate):
                frame_results = adapter.track_video_path(
                    sam3_video_path,
                    prompt=args.prompt,
                    fps=metadata.fps,
                    max_frames=args.max_frames,
                )
            materialize_frame_result_contours(
                frame_results,
                include_contours=need_contours,
                contour_epsilon_px=args.contour_epsilon_px,
            )

        frame_results = apply_source_frame_timestamps(
            frame_results,
            source_frame_timestamps,
        )

        annotated_video_path = None
        if should_write_video:
            annotated_video_path = output_dir / f"{video_path.stem}_sam3_contours.mp4"
            draw_tracked_video(
                source_video_path=draw_source_video_path,
                output_path=annotated_video_path,
                frame_results=frame_results,
                draw_boxes=args.draw_boxes,
                draw_contours=args.draw_contours or not args.draw_boxes,
                roi=roi,
                output_fps=None if processing_video_path is not None else processing_fps,
                frame_result_fps=(
                    metadata.fps if processing_video_path is not None else None
                ),
            )

        serialized_frames = []
        include_contours = contour_json_requested(args)
        for frame_result in frame_results:
            tracks = (
                offset_tracks(frame_result.tracks, roi) if roi else frame_result.tracks
            )
            serialized_frames.append(
                {
                    "frame_index": frame_result.frame_index,
                    "timestamp_seconds": frame_result.timestamp_seconds,
                    "tracks": serialize_tracks(
                        tracks,
                        include_contours=include_contours,
                        contour_epsilon_px=args.contour_epsilon_px,
                    ),
                }
            )
            strip_track_masks(frame_result.tracks)

        result = {
            "input_type": "video",
            "video": str(metadata.path),
            "model": "sam3",
            "video_mode": "whole",
            "roi_xyxy": roi,
            "sam3_video_path": str(sam3_video_path) if roi else None,
            "annotated_video_path": (
                str(annotated_video_path) if annotated_video_path else None
            ),
            "frames": serialized_frames,
        }
        if chunks is not None:
            result["split"] = {
                "split_seconds": args.split_seconds,
                "overlap_seconds": args.overlap_seconds,
                "chunks": serialize_sam3_video_chunks(chunks),
            }
        emit_json_result(result, args)
        timer.print_summary()
    return 0


def crop_video_to_roi(
    video_path: Path,
    output_dir: Path,
    roi: ROI,
) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "Whole-video ROI mode requires ffmpeg. Install it with "
            "`sudo apt install -y ffmpeg`."
        )
    x1, y1, x2, y2 = roi
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid ROI: {roi}")
    output_path = output_dir / f"{video_path.stem}_roi_sam3_input.mp4"
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"crop={width}:{height}:{x1}:{y1}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"ffmpeg failed while cropping ROI video: {stderr}")
    return output_path


def prepare_sam3_processing_video(
    video_path: Path,
    output_dir: Path,
    *,
    fps: float,
    max_frames: int | None = None,
) -> Path:
    """Create the one constant-FPS video used by both SAM3 and rendering."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "Sampled whole-video SAM3 mode requires ffmpeg. Install it with "
            "`sudo apt install -y ffmpeg`."
        )
    if fps <= 0:
        raise ValueError("SAM3 processing fps must be positive.")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("--max-frames must be positive when set.")

    output_dir.mkdir(parents=True, exist_ok=True)
    fps_token = f"{fps:g}".replace(".", "_")
    output_path = output_dir / f"{video_path.stem}_sam3_{fps_token}fps.mp4"
    frame_limit_args = (
        ["-frames:v", str(max_frames)] if max_frames is not None else []
    )
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        (
            f"fps=fps={fps:g}:round=near:start_time=0,"
            f"setpts=N/({fps:g}*TB)"
        ),
        *frame_limit_args,
        "-r",
        f"{fps:g}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(
            f"ffmpeg failed while creating the SAM3 processing video: {stderr}"
        )
    return output_path


def build_sam3_video_chunk_specs(
    video_path: Path,
    output_dir: Path,
    *,
    duration_seconds: float,
    fps: float,
    split_seconds: float,
    overlap_seconds: float,
    max_frames: int | None = None,
    processing_fps: float | None = None,
    source_frame_timestamps: list[float] | None = None,
) -> list[Sam3VideoChunk]:
    if fps <= 0:
        raise ValueError("Video fps must be positive.")
    if split_seconds <= 0:
        raise ValueError("--split-seconds must be positive.")
    if overlap_seconds < 0:
        raise ValueError("--overlap-seconds must be non-negative.")
    if overlap_seconds >= split_seconds:
        raise ValueError("--overlap-seconds must be less than --split-seconds.")
    if processing_fps is not None and processing_fps <= 0:
        raise ValueError("--sample-fps must be positive when set.")
    if source_frame_timestamps is not None and any(
        current < previous
        for previous, current in zip(
            source_frame_timestamps,
            source_frame_timestamps[1:],
        )
    ):
        raise ValueError("Source frame timestamps must be sorted.")

    chunk_fps = processing_fps or fps

    effective_duration = duration_seconds
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("--max-frames must be positive when set.")
        effective_duration = min(effective_duration, max_frames / chunk_fps)
    if effective_duration <= 0:
        return []

    chunks = []
    step_seconds = split_seconds - overlap_seconds
    start_seconds = 0.0
    index = 0
    while start_seconds < effective_duration - 1e-6:
        chunk_duration = min(split_seconds, effective_duration - start_seconds)
        start_frame_index = frame_index_at_or_after_timestamp(
            source_frame_timestamps,
            start_seconds,
            fallback_fps=fps,
        )
        emit_start_seconds = (
            start_seconds if index == 0 else start_seconds + overlap_seconds
        )
        emit_start_frame_index = frame_index_at_or_after_timestamp(
            source_frame_timestamps,
            emit_start_seconds,
            fallback_fps=fps,
        )
        if processing_fps is None:
            end_frame_index = frame_index_at_or_after_timestamp(
                source_frame_timestamps,
                start_seconds + chunk_duration,
                fallback_fps=fps,
            )
            chunk_max_frames = max(1, end_frame_index - start_frame_index)
        else:
            chunk_max_frames = seconds_to_frame_count(chunk_duration, chunk_fps)
        if max_frames is not None:
            start_sample_index = (
                start_frame_index
                if processing_fps is None and source_frame_timestamps is not None
                else seconds_to_frame_index(start_seconds, chunk_fps)
            )
            chunk_max_frames = min(chunk_max_frames, max_frames - start_sample_index)
        if chunk_max_frames <= 0:
            break

        chunks.append(
            Sam3VideoChunk(
                index=index,
                path=output_dir / f"{video_path.stem}_sam3_chunk_{index:04d}.mp4",
                start_seconds=start_seconds,
                duration_seconds=chunk_duration,
                start_frame_index=start_frame_index,
                emit_start_frame_index=emit_start_frame_index,
                max_frames=chunk_max_frames,
                processing_fps=processing_fps,
            )
        )
        if start_seconds + split_seconds >= effective_duration - 1e-6:
            break
        start_seconds += step_seconds
        index += 1
    return chunks


def serialize_sam3_video_chunks(chunks: list[Sam3VideoChunk]) -> list[dict]:
    return [
        {
            "index": chunk.index,
            "path": str(chunk.path),
            "start_seconds": chunk.start_seconds,
            "duration_seconds": chunk.duration_seconds,
            "start_frame_index": chunk.start_frame_index,
            "emit_start_frame_index": chunk.emit_start_frame_index,
            "max_frames": chunk.max_frames,
            "processing_fps": chunk.processing_fps,
        }
        for chunk in chunks
    ]


def seconds_to_frame_index(seconds: float, fps: float) -> int:
    return max(0, int(round(seconds * fps)))


def frame_index_at_or_after_timestamp(
    frame_timestamps: list[float] | None,
    timestamp_seconds: float,
    *,
    fallback_fps: float,
) -> int:
    if frame_timestamps is None:
        return seconds_to_frame_index(timestamp_seconds, fallback_fps)
    return bisect_left(frame_timestamps, timestamp_seconds)


def probe_video_frame_timestamps(video_path: Path) -> list[float]:
    """Return presentation timestamps normalized to the video's first frame."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError(
            "Whole-video SAM3 mode requires ffprobe. Install it with "
            "`sudo apt install -y ffmpeg`."
        )
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "frame=best_effort_timestamp_time",
        "-of",
        "csv=p=0",
        str(video_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"ffprobe failed while reading frame timestamps: {stderr}")
    timestamps = []
    for line in completed.stdout.splitlines():
        timestamp_token = line.partition(",")[0].strip()
        if not timestamp_token or timestamp_token == "N/A":
            continue
        timestamps.append(float(timestamp_token))
    if not timestamps:
        raise RuntimeError(f"ffprobe returned no video frame timestamps for {video_path}")
    first_timestamp = timestamps[0]
    return [timestamp - first_timestamp for timestamp in timestamps]


def apply_source_frame_timestamps(
    frame_results: list[Sam3FrameResult],
    source_frame_timestamps: list[float] | None,
) -> list[Sam3FrameResult]:
    """Replace nominal-FPS timestamps with source presentation timestamps."""
    if source_frame_timestamps is None:
        return frame_results

    remapped_results = []
    for frame_result in frame_results:
        frame_index = frame_result.frame_index
        if (
            frame_index is None
            or frame_index < 0
            or frame_index >= len(source_frame_timestamps)
        ):
            remapped_results.append(frame_result)
            continue

        timestamp_seconds = source_frame_timestamps[frame_index]
        remapped_results.append(
            replace(
                frame_result,
                timestamp_seconds=timestamp_seconds,
                tracks=[
                    replace(track, timestamp_seconds=timestamp_seconds)
                    for track in frame_result.tracks
                ],
            )
        )
    return remapped_results


def seconds_to_frame_count(seconds: float, fps: float) -> int:
    return max(1, int(math.ceil(seconds * fps - 1e-9)))


def resolve_sam3_chunk_processing_fps(
    args: argparse.Namespace,
    source_fps: float,
) -> float | None:
    if args.split_seconds is None or args.sample_fps is None:
        return None
    if args.sample_fps > source_fps:
        print(
            f"warning: --sample-fps {args.sample_fps:g} is above source fps "
            f"{source_fps:.2f}; using source fps",
            file=sys.stderr,
        )
        return source_fps
    return args.sample_fps


def chunk_processing_fps(chunk: Sam3VideoChunk, source_fps: float) -> float:
    return chunk.processing_fps or source_fps


def split_sam3_video_chunks(
    video_path: Path,
    chunks: list[Sam3VideoChunk],
    *,
    input_fps: float | None = None,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "Split whole-video SAM3 mode requires ffmpeg. Install it with "
            "`sudo apt install -y ffmpeg`."
        )
    for chunk in chunks:
        chunk.path.parent.mkdir(parents=True, exist_ok=True)
        video_filter_args = []
        if chunk.processing_fps is not None and (
            input_fps is None
            or not math.isclose(
                input_fps,
                chunk.processing_fps,
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
        ):
            video_filter_args = ["-vf", f"fps={chunk.processing_fps:g}"]
        frame_limit_args = (
            ["-frames:v", str(chunk.max_frames)]
            if chunk.max_frames is not None
            else []
        )
        command = [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-ss",
            f"{chunk.start_seconds:.6f}",
            "-t",
            f"{chunk.duration_seconds:.6f}",
            "-i",
            str(video_path),
            "-map",
            "0:v:0",
            "-an",
            *video_filter_args,
            *frame_limit_args,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(chunk.path),
        ]
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode != 0:
            stderr = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"ffmpeg failed while creating SAM3 chunk {chunk.index}: {stderr}"
            )


def track_sam3_video_chunks(
    adapter: SAM3ModelAdapter,
    chunks: list[Sam3VideoChunk],
    *,
    prompt: str | list[str] | None,
    fps: float,
) -> list[list[Sam3FrameResult]]:
    chunk_results = []
    total_chunks = len(chunks)
    for chunk in chunks:
        processing_fps = chunk_processing_fps(chunk, fps)
        print(
            f"sam3_chunk: {chunk.index + 1}/{total_chunks} "
            f"start={chunk.start_seconds:.2f}s "
            f"duration={chunk.duration_seconds:.2f}s "
            f"fps={processing_fps:.2f}"
        )
        chunk_results.append(
            adapter.track_video_path(
                chunk.path,
                prompt=prompt,
                fps=processing_fps,
                max_frames=chunk.max_frames,
            )
        )
        clear_cuda_cache()
    return chunk_results


def track_and_merge_sam3_video_chunks(
    adapter: SAM3ModelAdapter,
    chunks: list[Sam3VideoChunk],
    *,
    prompt: str | list[str] | None,
    fps: float,
    materialize_contours: bool = True,
    contour_epsilon_px: float = 2.0,
    match_iou_threshold: float = SAM3_SPLIT_TRACK_MATCH_IOU_THRESHOLD,
) -> list[Sam3FrameResult]:
    accepted_by_frame: dict[int, Sam3FrameResult] = {}
    used_global_track_ids: set[str] = set()
    total_chunks = len(chunks)

    for chunk in chunks:
        processing_fps = chunk_processing_fps(chunk, fps)
        print(
            f"sam3_chunk: {chunk.index + 1}/{total_chunks} "
            f"start={chunk.start_seconds:.2f}s "
            f"duration={chunk.duration_seconds:.2f}s "
            f"fps={processing_fps:.2f}",
            flush=True,
        )
        local_results = adapter.track_video_path(
            chunk.path,
            prompt=prompt,
            fps=processing_fps,
            max_frames=chunk.max_frames,
        )
        materialize_frame_result_contours(
            local_results,
            include_contours=materialize_contours,
            contour_epsilon_px=contour_epsilon_px,
        )
        merge_sam3_chunk_result(
            chunk,
            local_results,
            accepted_by_frame,
            used_global_track_ids,
            fps=fps,
            match_iou_threshold=match_iou_threshold,
        )
        del local_results
        clear_cuda_cache()

    return [accepted_by_frame[index] for index in sorted(accepted_by_frame)]


def clear_cuda_cache() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def merge_sam3_chunk_results(
    chunks: list[Sam3VideoChunk],
    chunk_results: list[list[Sam3FrameResult]],
    *,
    fps: float,
    match_iou_threshold: float = SAM3_SPLIT_TRACK_MATCH_IOU_THRESHOLD,
) -> list[Sam3FrameResult]:
    accepted_by_frame: dict[int, Sam3FrameResult] = {}
    used_global_track_ids: set[str] = set()

    for chunk, local_results in zip(chunks, chunk_results):
        merge_sam3_chunk_result(
            chunk,
            local_results,
            accepted_by_frame,
            used_global_track_ids,
            fps=fps,
            match_iou_threshold=match_iou_threshold,
        )

    return [accepted_by_frame[index] for index in sorted(accepted_by_frame)]


def merge_sam3_chunk_result(
    chunk: Sam3VideoChunk,
    local_results: list[Sam3FrameResult],
    accepted_by_frame: dict[int, Sam3FrameResult],
    used_global_track_ids: set[str],
    *,
    fps: float,
    match_iou_threshold: float,
) -> None:
    local_to_global = match_chunk_track_ids(
        chunk,
        local_results,
        accepted_by_frame,
        fps=fps,
        match_iou_threshold=match_iou_threshold,
    )
    has_stitched_track = bool(local_to_global)
    reserved_global_track_ids = set(used_global_track_ids) | set(
        local_to_global.values()
    )
    for track_id in sorted(local_track_ids(local_results)):
        if track_id not in local_to_global:
            local_to_global[track_id] = allocate_global_track_id(
                track_id,
                chunk.index,
                reserved_global_track_ids,
            )
            reserved_global_track_ids.add(local_to_global[track_id])
    used_global_track_ids.update(reserved_global_track_ids)

    for local_result in local_results:
        global_result = globalize_chunk_frame_result(
            chunk,
            local_result,
            fps=fps,
            local_to_global=local_to_global,
        )
        if global_result.frame_index is None:
            continue
        if global_result.frame_index < chunk.start_frame_index:
            continue
        if global_result.frame_index < chunk.emit_start_frame_index:
            if global_result.frame_index not in accepted_by_frame:
                continue
            if not has_stitched_track:
                continue
        # ID matching above compares this chunk's overlap with the previously
        # accepted chunk. After a successful match, prefer this newer chunk's
        # masks in the overlap because they were propagated from a newer prompt.
        accepted_by_frame[global_result.frame_index] = global_result


def local_track_ids(frame_results: list[Sam3FrameResult]) -> set[str]:
    return {
        track.track_id
        for frame_result in frame_results
        for track in frame_result.tracks
        if track.state == "active"
    }


def match_chunk_track_ids(
    chunk: Sam3VideoChunk,
    local_results: list[Sam3FrameResult],
    accepted_by_frame: dict[int, Sam3FrameResult],
    *,
    fps: float,
    match_iou_threshold: float,
) -> dict[str, str]:
    if not accepted_by_frame or chunk.emit_start_frame_index <= chunk.start_frame_index:
        return {}

    scores: dict[tuple[str, str], list[float]] = {}
    for local_result in local_results:
        global_frame_index = global_frame_index_for_chunk_result(
            chunk,
            local_result,
            source_fps=fps,
        )
        if global_frame_index < chunk.start_frame_index:
            continue
        if global_frame_index >= chunk.emit_start_frame_index:
            continue
        previous_result = accepted_by_frame.get(global_frame_index)
        if previous_result is None:
            continue
        for local_track in active_tracks(local_result.tracks):
            for global_track in active_tracks(previous_result.tracks):
                if local_track.label.lower() != global_track.label.lower():
                    continue
                overlap = track_bbox_iou(local_track, global_track)
                if overlap <= 0:
                    continue
                scores.setdefault(
                    (local_track.track_id, global_track.track_id),
                    [],
                ).append(overlap)

    ranked_pairs = sorted(
        (
            (sum(values) / len(values), local_id, global_id)
            for (local_id, global_id), values in scores.items()
            if values
        ),
        reverse=True,
    )
    local_to_global = {}
    used_global_ids = set()
    for score, local_id, global_id in ranked_pairs:
        if score < match_iou_threshold:
            continue
        if local_id in local_to_global or global_id in used_global_ids:
            continue
        local_to_global[local_id] = global_id
        used_global_ids.add(global_id)
    return local_to_global


def global_frame_index_for_chunk_result(
    chunk: Sam3VideoChunk,
    frame_result: Sam3FrameResult,
    *,
    source_fps: float,
) -> int:
    if frame_result.frame_index is not None:
        processing_fps = chunk_processing_fps(chunk, source_fps)
        source_frame_offset = seconds_to_frame_index(
            int(frame_result.frame_index) / processing_fps,
            source_fps,
        )
        return chunk.start_frame_index + source_frame_offset
    return seconds_to_frame_index(
        chunk.start_seconds + frame_result.timestamp_seconds,
        source_fps,
    )


def global_timestamp_seconds_for_chunk_result(
    chunk: Sam3VideoChunk,
    frame_result: Sam3FrameResult,
    *,
    source_fps: float,
) -> float:
    if frame_result.frame_index is not None:
        return global_frame_index_for_chunk_result(
            chunk,
            frame_result,
            source_fps=source_fps,
        ) / source_fps
    return chunk.start_seconds + frame_result.timestamp_seconds


def active_tracks(tracks: list[Track]) -> list[Track]:
    return [track for track in tracks if track.state == "active"]


def allocate_global_track_id(
    local_track_id: str,
    chunk_index: int,
    used_global_track_ids: set[str],
) -> str:
    if local_track_id not in used_global_track_ids:
        return local_track_id
    namespace, local_index = split_track_id_namespace(local_track_id)
    if namespace is None:
        namespace = "global"
        local_index = chunk_index

    next_index = max(
        local_index + 1,
        next_track_id_index(namespace, used_global_track_ids),
    )
    candidate = f"{namespace}:{next_index}"
    while candidate in used_global_track_ids:
        next_index += 1
        candidate = f"{namespace}:{next_index}"
    return candidate


def split_track_id_namespace(track_id: str) -> tuple[str | None, int]:
    match = re.fullmatch(r"(.+):(\d+)", str(track_id))
    if match is None:
        return None, 0
    return match.group(1), int(match.group(2))


def next_track_id_index(namespace: str, used_global_track_ids: set[str]) -> int:
    next_index = 0
    prefix = f"{namespace}:"
    for track_id in used_global_track_ids:
        if not track_id.startswith(prefix):
            continue
        _, index = split_track_id_namespace(track_id)
        next_index = max(next_index, index + 1)
    return next_index


def globalize_chunk_frame_result(
    chunk: Sam3VideoChunk,
    local_result: Sam3FrameResult,
    *,
    fps: float,
    local_to_global: dict[str, str],
) -> Sam3FrameResult:
    global_frame_index = global_frame_index_for_chunk_result(
        chunk,
        local_result,
        source_fps=fps,
    )
    timestamp_seconds = global_timestamp_seconds_for_chunk_result(
        chunk,
        local_result,
        source_fps=fps,
    )
    tracks = [
        replace(
            track,
            track_id=local_to_global.get(track.track_id, track.track_id),
            timestamp_seconds=timestamp_seconds,
        )
        for track in local_result.tracks
    ]
    return Sam3FrameResult(
        timestamp_seconds=timestamp_seconds,
        image_path=local_result.image_path,
        tracks=tracks,
        frame_index=global_frame_index,
    )


def track_bbox_iou(track_a: Track, track_b: Track) -> float:
    bbox_a = track_a.bbox
    bbox_b = track_b.bbox
    x1 = max(bbox_a.x1, bbox_b.x1)
    y1 = max(bbox_a.y1, bbox_b.y1)
    x2 = min(bbox_a.x2, bbox_b.x2)
    y2 = min(bbox_a.y2, bbox_b.y2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if intersection <= 0:
        return 0.0
    union = bbox_a.width * bbox_a.height + bbox_b.width * bbox_b.height - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def run_sam3_live_file(
    video_path: Path,
    live_config: Sam3LiveConfig,
    model_config_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    if not video_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {video_path}")

    metadata = probe_video(video_path)
    print(
        f"video: {metadata.path} "
        f"{metadata.width}x{metadata.height} "
        f"{metadata.duration_seconds:.2f}s "
        f"{metadata.fps:.2f}fps"
    )
    start_from_seconds = args.start_from
    duration = max(0.0, metadata.duration_seconds - start_from_seconds)
    if args.max_runtime_seconds is not None:
        duration = min(duration, args.max_runtime_seconds)
    max_frames = max(1, int(duration * live_config.sample_fps) + 1)
    end_at_seconds = start_from_seconds + duration
    source_frames = extract_frames_at_fps(
        video_path,
        output_dir,
        sample_fps=live_config.sample_fps,
        max_frames=max_frames,
        start_from_seconds=start_from_seconds,
        worker_threads=args.number_worker_threads,
    )
    if args.max_runtime_seconds is not None:
        source_frames = [
            frame
            for frame in source_frames
            if frame.timestamp_seconds <= end_at_seconds
        ]
    sam_frames = prepare_sam3_live_frames(source_frames, output_dir, live_config.roi)
    windows = build_frame_windows(sam_frames, live_config)
    if args.max_windows is not None:
        windows = windows[: args.max_windows]
    print(f"sampled_frames: {len(sam_frames)}")
    print(f"windows: {len(windows)}")

    if args.extract_only:
        for window in windows:
            print_window_summary(window)
        return {
            "input_type": "video",
            "source": str(video_path),
            "model": "sam3-live",
            "strategy": live_config.strategy,
            "roi_xyxy": live_config.roi,
            "sampled_frames": len(sam_frames),
            "windows": [serialize_window(window) for window in windows],
            "events": [],
        }

    return process_sam3_live_windows(
        source=str(video_path),
        source_type="file",
        windows=windows,
        live_config=live_config,
        model_config_path=model_config_path,
        output_dir=output_dir,
        args=args,
    )


def run_sam3_live_rtsp(
    source_uri: str,
    live_config: Sam3LiveConfig,
    model_config_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    sampler = RTSPFrameSampler(
        source_uri=source_uri,
        output_dir=output_dir / "rtsp_frames",
        sample_fps=live_config.sample_fps,
        roi=live_config.roi,
        crop_output_dir=output_dir,
    )
    stitcher = Sam3TrackStitcher(live_config)
    adapter = None if args.extract_only else SAM3ModelAdapter(
        SAM3Config.from_mapping(load_yaml(model_config_path))
    )
    timer = LatencyTimer(enabled=args.timeit)
    result = base_sam3_live_result(
        source=source_uri,
        source_type="rtsp",
        live_config=live_config,
    )
    evidence_frames: list[Path] = []
    frame_buffer: list[SampledFrame] = []
    last_committed_seconds = float("-inf")
    window_index = 0
    next_window_end = live_config.window_seconds
    overlap_seconds = max(0.0, live_config.window_seconds - live_config.stride_seconds)

    try:
        while args.max_windows is None or window_index < args.max_windows:
            if args.max_runtime_seconds is not None and (
                next_window_end > args.max_runtime_seconds
            ):
                break
            new_frames = sampler.capture_until(next_window_end)
            frame_buffer.extend(new_frames)
            window_start = max(0.0, next_window_end - live_config.window_seconds)
            window_frames = [
                frame
                for frame in frame_buffer
                if window_start <= frame.timestamp_seconds <= next_window_end
            ]
            if not window_frames:
                next_window_end += live_config.stride_seconds
                continue

            window = FrameWindow(
                window_index=window_index,
                frames=window_frames,
                commit_after_seconds=last_committed_seconds,
            )
            print_window_summary(window)
            result["windows"].append(serialize_window(window))
            if not args.extract_only and adapter is not None:
                window_result = process_one_sam3_live_window(
                    adapter,
                    stitcher,
                    window,
                    live_config,
                    timer,
                )
                append_sam3_live_window_result(
                    result,
                    window_result,
                    args,
                    evidence_frames,
                )
                if window.frames:
                    last_committed_seconds = window.frames[-1].timestamp_seconds

            keep_after = max(0.0, next_window_end - overlap_seconds)
            frame_buffer = [
                frame for frame in frame_buffer if frame.timestamp_seconds >= keep_after
            ]
            window_index += 1
            next_window_end += live_config.stride_seconds
    finally:
        sampler.close()

    evidence_video_path = maybe_write_evidence_video(
        evidence_frames,
        output_dir,
        live_config,
        args,
    )
    result["evidence_video_path"] = (
        str(evidence_video_path) if evidence_video_path else None
    )
    timer.print_summary()
    return result


def run_sam3_live_rtsp_smoke_test(
    source_uri: str,
    live_config: Sam3LiveConfig,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    frame_count = max(1, args.smoke_test_frames)
    sampler = RTSPFrameSampler(
        source_uri=source_uri,
        output_dir=output_dir / "rtsp_smoke_frames",
        sample_fps=live_config.sample_fps,
        roi=live_config.roi,
        crop_output_dir=output_dir,
    )
    started_at = time.perf_counter()
    try:
        frames = sampler.capture_count(frame_count)
    finally:
        sampler.close()

    elapsed_seconds = time.perf_counter() - started_at
    return {
        "input_type": "rtsp",
        "source": source_uri,
        "model": "sam3-live",
        "smoke_test": True,
        "status": "ok",
        "roi_xyxy": live_config.roi,
        "sample_fps": live_config.sample_fps,
        "requested_frames": frame_count,
        "captured_frames": len(frames),
        "elapsed_seconds": elapsed_seconds,
        "frames": [
            {
                "timestamp_seconds": frame.timestamp_seconds,
                "image_path": str(frame.path),
            }
            for frame in frames
        ],
    }


def process_sam3_live_windows(
    source: str,
    source_type: str,
    windows: list[FrameWindow],
    live_config: Sam3LiveConfig,
    model_config_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict:
    adapter = SAM3ModelAdapter(SAM3Config.from_mapping(load_yaml(model_config_path)))
    stitcher = Sam3TrackStitcher(live_config)
    timer = LatencyTimer(enabled=args.timeit)
    result = base_sam3_live_result(
        source=source,
        source_type=source_type,
        live_config=live_config,
    )
    evidence_frames: list[Path] = []

    for window in windows:
        print_window_summary(window)
        result["windows"].append(serialize_window(window))
        window_result = process_one_sam3_live_window(
            adapter,
            stitcher,
            window,
            live_config,
            timer,
        )
        append_sam3_live_window_result(
            result,
            window_result,
            args,
            evidence_frames,
        )

    evidence_video_path = maybe_write_evidence_video(
        evidence_frames,
        output_dir,
        live_config,
        args,
    )
    result["evidence_video_path"] = (
        str(evidence_video_path) if evidence_video_path else None
    )
    timer.print_summary()
    return result


def process_one_sam3_live_window(
    adapter: SAM3ModelAdapter,
    stitcher: Sam3TrackStitcher,
    window: FrameWindow,
    live_config: Sam3LiveConfig,
    timer: LatencyTimer,
) -> StitchWindowResult:
    with timer.measure(frame_count=len(window.frames)):
        if live_config.strategy == "per_frame":
            frame_results = [
                Sam3FrameResult(
                    timestamp_seconds=frame.timestamp_seconds,
                    image_path=frame.path,
                    tracks=detections_to_ephemeral_tracks(
                        adapter.detect_image_path(
                            frame.path,
                            prompt=live_config.prompt,
                            timestamp_seconds=frame.timestamp_seconds,
                        ),
                        frame.timestamp_seconds,
                    ),
                )
                for frame in window.frames
            ]
        else:
            frame_results = adapter.track_video_frames(
                window.frames,
                prompt=live_config.prompt,
            )

    return stitcher.process_window(
        window.window_index,
        frame_results,
        commit_after_seconds=window.commit_after_seconds,
    )


def append_sam3_live_window_result(
    result: dict,
    window_result: StitchWindowResult,
    args: argparse.Namespace,
    evidence_frames: list[Path],
) -> None:
    serialized_frames = []
    for frame_result in window_result.committed_frames:
        annotated_image_path = None
        if args.draw_boxes:
            annotated_image_path = frame_result.image_path.with_name(
                f"{frame_result.image_path.stem}_global_boxes"
                f"{frame_result.image_path.suffix}"
            )
            draw_tracks(frame_result.image_path, annotated_image_path, frame_result.tracks)
            evidence_frames.append(annotated_image_path)
        serialized_frames.append(
            {
                "timestamp_seconds": frame_result.timestamp_seconds,
                "image_path": str(frame_result.image_path),
                "annotated_image_path": (
                    str(annotated_image_path) if annotated_image_path else None
                ),
                "tracks": serialize_tracks(frame_result.tracks),
            }
        )

    result["committed_frames"].extend(serialized_frames)
    result["events"].extend(event.as_dict() for event in window_result.events)


def prepare_sam3_live_frames(
    source_frames: list[SampledFrame],
    output_dir: Path,
    roi: ROI | None,
) -> list[SampledFrame]:
    if roi is None:
        return source_frames
    return crop_frames_to_roi(source_frames, output_dir, roi)


def detections_to_ephemeral_tracks(
    detections: list[Detection],
    timestamp_seconds: float,
) -> list[Track]:
    return [
        Track(
            track_id=f"det-{index}",
            bbox=detection.bbox,
            label=detection.label,
            confidence=detection.confidence,
            timestamp_seconds=timestamp_seconds,
            metadata=detection.metadata,
        )
        for index, detection in enumerate(detections)
    ]


def base_sam3_live_result(
    source: str,
    source_type: str,
    live_config: Sam3LiveConfig,
) -> dict:
    return {
        "input_type": source_type,
        "source": source,
        "model": "sam3-live",
        "strategy": live_config.strategy,
        "prompt": list(live_config.prompt),
        "roi_xyxy": live_config.roi,
        "sample_fps": live_config.sample_fps,
        "window_seconds": live_config.window_seconds,
        "stride_seconds": live_config.stride_seconds,
        "dwell_threshold_seconds": live_config.dwell_threshold_seconds,
        "missing_grace_seconds": live_config.missing_grace_seconds,
        "coordinates": "roi_crop" if live_config.roi else "source_frame",
        "windows": [],
        "committed_frames": [],
        "events": [],
        "evidence_video_path": None,
    }


def print_window_summary(window: FrameWindow) -> None:
    commit_after = finite_float_or_none(window.commit_after_seconds)
    commit_text = "none" if commit_after is None else f"{commit_after:.2f}s"
    print(
        f"window-{window.window_index}: "
        f"{window.start_seconds:.2f}s-{window.end_seconds:.2f}s "
        f"frames={len(window.frames)} "
        f"commit_after={commit_text}"
    )


def serialize_window(window: FrameWindow) -> dict:
    return {
        "window_index": window.window_index,
        "start_seconds": window.start_seconds,
        "end_seconds": window.end_seconds,
        "commit_after_seconds": finite_float_or_none(window.commit_after_seconds),
        "frame_count": len(window.frames),
        "frame_paths": [str(frame.path) for frame in window.frames],
    }


def finite_float_or_none(value: float) -> float | None:
    if not math.isfinite(value):
        return None
    return value


def maybe_write_evidence_video(
    image_paths: list[Path],
    output_dir: Path,
    live_config: Sam3LiveConfig,
    args: argparse.Namespace,
) -> Path | None:
    if not args.evidence_video:
        return None
    if not image_paths:
        return None
    output_path = output_dir / "sam3_live_evidence.mp4"
    write_image_sequence_video(
        image_paths,
        output_path,
        fps=live_config.sample_fps,
    )
    return output_path


def write_image_sequence_video(
    image_paths: list[Path],
    output_path: Path,
    fps: float,
) -> None:
    if not image_paths:
        raise ValueError("image_paths must not be empty")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "Creating evidence videos requires ffmpeg. Install it with "
            "`sudo apt install -y ffmpeg`."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    list_path = output_path.with_suffix(".frames.txt")
    duration = 1.0 / max(fps, 1e-6)
    with list_path.open("w", encoding="utf-8") as handle:
        for image_path in image_paths:
            handle.write(f"file '{escape_ffconcat_path(image_path.resolve())}'\n")
            handle.write(f"duration {duration:.6f}\n")
        handle.write(f"file '{escape_ffconcat_path(image_paths[-1].resolve())}'\n")

    command = [
        ffmpeg,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_path),
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip()
        if len(stderr) > 2000:
            stderr = stderr[-2000:]
        raise RuntimeError(
            f"ffmpeg failed while creating {output_path} "
            f"(exit {completed.returncode}):\n{stderr}"
        )


def escape_ffconcat_path(path: Path) -> str:
    """Escape a path for a quoted ffmpeg concat-demuxer file entry."""
    return str(path).replace("\\", "\\\\").replace("'", "'\\''")


class RTSPFrameSampler:
    """Single-process RTSP sampler for low-FPS SAM window smoke tests."""

    def __init__(
        self,
        source_uri: str,
        output_dir: Path,
        sample_fps: float,
        roi: ROI | None,
        crop_output_dir: Path,
    ) -> None:
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError(
                "RTSP sampling requires OpenCV. Install it with "
                "`pip install opencv-python`."
            ) from exc

        self.cv2 = cv2
        self.capture = cv2.VideoCapture(source_uri)
        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open RTSP/video stream: {source_uri}")
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_interval_seconds = 1.0 / sample_fps
        self.roi = roi
        self.crop_output_dir = crop_output_dir
        self.next_sample_timestamp_seconds = 0.0
        self.frame_index = 0

    def capture_until(self, end_timestamp_seconds: float) -> list[SampledFrame]:
        frames: list[SampledFrame] = []
        while self.next_sample_timestamp_seconds <= end_timestamp_seconds:
            ok, frame = self.capture.read()
            if not ok:
                raise RuntimeError("RTSP stream ended or could not read a frame.")
            timestamp_seconds = self.next_sample_timestamp_seconds

            frames.append(self._write_sampled_frame(frame, timestamp_seconds))
            self.frame_index += 1
            self.next_sample_timestamp_seconds += self.sample_interval_seconds
            if self.next_sample_timestamp_seconds <= end_timestamp_seconds:
                time.sleep(self.sample_interval_seconds)
        return frames

    def capture_count(self, frame_count: int) -> list[SampledFrame]:
        frames: list[SampledFrame] = []
        for _ in range(frame_count):
            ok, frame = self.capture.read()
            if not ok:
                raise RuntimeError("RTSP stream ended or could not read a frame.")
            timestamp_seconds = self.next_sample_timestamp_seconds
            frames.append(self._write_sampled_frame(frame, timestamp_seconds))
            self.frame_index += 1
            self.next_sample_timestamp_seconds += self.sample_interval_seconds
        return frames

    def close(self) -> None:
        self.capture.release()

    def _write_sampled_frame(
        self,
        frame: object,
        timestamp_seconds: float,
    ) -> SampledFrame:
        frame_path = self.output_dir / (
            f"rtsp_{self.frame_index:06d}_{timestamp_seconds:.2f}s.jpg"
        )
        if not self.cv2.imwrite(str(frame_path), frame):
            raise RuntimeError(f"Could not write RTSP frame: {frame_path}")
        sampled_frame = SampledFrame(
            path=frame_path,
            timestamp_seconds=timestamp_seconds,
        )
        if self.roi:
            sampled_frame = SampledFrame(
                path=crop_image_to_roi(
                    frame_path,
                    self.crop_output_dir,
                    self.roi,
                ),
                timestamp_seconds=timestamp_seconds,
            )
        return sampled_frame


def run_qwen3_vl_image(image_path: Path, args: argparse.Namespace) -> int:
    print(f"image: {image_path}")

    if args.extract_only:
        print(f"0.00s {image_path}")
        return 0

    timer = LatencyTimer(enabled=args.timeit)
    client = build_qwen3_vl_client(args.model_config, args.endpoint)
    with timer.measure():
        frame_result = run_qwen3_vl_on_frame(
            client,
            image_path=image_path,
            timestamp_seconds=0.0,
            args=args,
        )
    annotated_image_path = None
    if args.draw_boxes and frame_result["detections"]:
        annotated_image_path = output_image_path(image_path, args.output_dir)
        draw_detections(image_path, annotated_image_path, frame_result["detections"])

    print(
        json.dumps(
            {
                "input_type": "image",
                "model": "qwen3-vl",
                "task": args.task,
                "image_path": str(image_path),
                "annotated_image_path": (
                    str(annotated_image_path) if annotated_image_path else None
                ),
                "prompt": frame_result["prompt"],
                "raw_result": frame_result["raw_result"],
                "detections": serialize_detections(frame_result["detections"]),
            },
            indent=2,
        )
    )
    timer.print_summary()
    return 0


def run_qwen3_vl_video(video_path: Path, args: argparse.Namespace) -> int:
    metadata = probe_video(video_path)
    print(
        f"video: {metadata.path} "
        f"{metadata.width}x{metadata.height} "
        f"{metadata.duration_seconds:.2f}s "
        f"{metadata.fps:.2f}fps"
    )

    with tempfile.TemporaryDirectory(prefix="vision_pipeline_qwen3_vl_") as temp_dir:
        output_dir = Path(args.output_dir) if args.output_dir else Path(temp_dir)
        if args.sample_fps is not None:
            frames = extract_frames_at_fps(
                video_path,
                output_dir,
                sample_fps=args.sample_fps,
                max_frames=args.max_frames,
                start_from_seconds=args.start_from,
                worker_threads=args.number_worker_threads,
            )
        else:
            frames = extract_sample_frames(
                video_path,
                output_dir,
                args.sample_count,
                start_from_seconds=args.start_from,
                worker_threads=args.number_worker_threads,
            )
        print(f"sampled_frames: {len(frames)}")

        if args.extract_only:
            for frame in frames:
                print(f"{frame.timestamp_seconds:.2f}s {frame.path}")
            return 0

        client = build_qwen3_vl_client(args.model_config, args.endpoint)
        timer = LatencyTimer(enabled=args.timeit)
        result = {
            "input_type": "video",
            "video": str(metadata.path),
            "model": "qwen3-vl",
            "task": args.task,
            "start_from_seconds": args.start_from,
            "frames": [],
        }

        for frame in frames:
            with timer.measure():
                frame_result = run_qwen3_vl_on_frame(
                    client,
                    image_path=frame.path,
                    timestamp_seconds=frame.timestamp_seconds,
                    args=args,
                )
            annotated_image_path = None
            if args.draw_boxes and frame_result["detections"]:
                annotated_image_path = frame.path.with_name(
                    f"{frame.path.stem}_boxes{frame.path.suffix}"
                )
                draw_detections(
                    frame.path,
                    annotated_image_path,
                    frame_result["detections"],
                )

            result["frames"].append(
                {
                    "timestamp_seconds": frame.timestamp_seconds,
                    "image_path": str(frame.path),
                    "annotated_image_path": (
                        str(annotated_image_path) if annotated_image_path else None
                    ),
                    "prompt": frame_result["prompt"],
                    "raw_result": frame_result["raw_result"],
                    "detections": serialize_detections(frame_result["detections"]),
                }
            )

        print(json.dumps(result, indent=2))
        timer.print_summary()
    return 0


class LatencyTimer:
    """Accumulate per-frame model/API latency for CLI reporting."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.total_seconds = 0.0
        self.frame_count = 0

    def measure(self, frame_count: int = 1) -> "LatencyMeasurement":
        return LatencyMeasurement(self, frame_count)

    def add_duration(self, seconds: float, frame_count: int) -> None:
        if not self.enabled or frame_count <= 0:
            return
        self.total_seconds += seconds
        self.frame_count += frame_count

    def print_summary(self) -> None:
        if not self.enabled:
            return
        if self.frame_count <= 0:
            print("timeit: frames=0 avg_frame_latency_ms=0.00 total_latency_ms=0.00")
            return
        average_ms = self.total_seconds / self.frame_count * 1000.0
        total_ms = self.total_seconds * 1000.0
        print(
            f"timeit: frames={self.frame_count} "
            f"avg_frame_latency_ms={average_ms:.2f} "
            f"total_latency_ms={total_ms:.2f}"
        )


class LatencyMeasurement:
    """Context manager used by LatencyTimer."""

    def __init__(self, timer: LatencyTimer, frame_count: int) -> None:
        self.timer = timer
        self.frame_count = frame_count
        self.started_at = 0.0

    def __enter__(self) -> None:
        if self.timer.enabled:
            self.started_at = time.perf_counter()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self.timer.enabled:
            elapsed_seconds = time.perf_counter() - self.started_at
            self.timer.add_duration(elapsed_seconds, self.frame_count)


@dataclass(frozen=True)
class TrackedFrameRecord:
    """Frame plus tracks observed at that sampled timestamp."""

    image_path: Path
    timestamp_seconds: float
    tracks: list[Track]


@dataclass(frozen=True)
class Sam3VideoChunk:
    """One temporary SAM3 whole-video chunk and its global frame mapping."""

    index: int
    path: Path
    start_seconds: float
    duration_seconds: float
    start_frame_index: int
    emit_start_frame_index: int
    max_frames: int | None
    processing_fps: float | None = None


@dataclass(frozen=True)
class TrackedFrameObservation:
    """One frame in a tracked-object clip, with an optional visible track."""

    image_path: Path
    timestamp_seconds: float
    track: Track | None


@dataclass(frozen=True)
class TrackAppearance:
    """A generated clip span for one tracked object."""

    track_id: str
    appearance_index: int
    observations: list[TrackedFrameObservation]


@dataclass
class _OpenTrackAppearance:
    track_id: str
    appearance_index: int
    observations: list[TrackedFrameObservation]
    last_frame_index: int


def resolve_video_output_dir(
    video_path: Path,
    args: argparse.Namespace,
    temp_dir: str,
    persistent_default: bool = False,
) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    if persistent_default:
        return Path.cwd() / f"{video_path.stem}_vision_pipeline_outputs"
    return Path(temp_dir)


def infer_frame_sequence_fps(
    frames: list[SampledFrame],
    fallback: float | None = None,
) -> float:
    if fallback is not None and fallback > 0:
        return fallback
    if len(frames) < 2:
        return 1.0

    ordered_timestamps = sorted(frame.timestamp_seconds for frame in frames)
    deltas = [
        later - earlier
        for earlier, later in zip(
            ordered_timestamps,
            ordered_timestamps[1:],
            strict=False,
        )
        if later > earlier
    ]
    if not deltas:
        return 1.0
    median_delta = sorted(deltas)[len(deltas) // 2]
    return 1.0 / max(median_delta, 1e-6)


def build_continuous_track_appearances(
    frame_records: list[TrackedFrameRecord],
) -> list[TrackAppearance]:
    ordered_records = sorted(
        frame_records,
        key=lambda item: item.timestamp_seconds,
    )
    completed: list[TrackAppearance] = []
    open_by_track_id: dict[str, _OpenTrackAppearance] = {}
    appearance_counts: dict[str, int] = {}

    for frame_index, record in enumerate(ordered_records):
        active_tracks = {
            track.track_id: track
            for track in record.tracks
            if track.state == "active"
        }
        for track_id, track in active_tracks.items():
            observation = TrackedFrameObservation(
                image_path=record.image_path,
                timestamp_seconds=record.timestamp_seconds,
                track=track,
            )
            open_appearance = open_by_track_id.get(track_id)
            if (
                open_appearance is not None
                and open_appearance.last_frame_index == frame_index - 1
            ):
                open_appearance.observations.append(observation)
                open_appearance.last_frame_index = frame_index
                continue

            if open_appearance is not None and open_appearance.observations:
                completed.append(
                    TrackAppearance(
                        track_id=track_id,
                        appearance_index=open_appearance.appearance_index,
                        observations=list(open_appearance.observations),
                    )
                )

            appearance_counts[track_id] = appearance_counts.get(track_id, 0) + 1
            open_by_track_id[track_id] = _OpenTrackAppearance(
                track_id=track_id,
                appearance_index=appearance_counts[track_id],
                observations=[observation],
                last_frame_index=frame_index,
            )

    for open_appearance in open_by_track_id.values():
        if open_appearance.observations:
            completed.append(
                TrackAppearance(
                    track_id=open_appearance.track_id,
                    appearance_index=open_appearance.appearance_index,
                    observations=list(open_appearance.observations),
                )
            )

    return sorted(
        completed,
        key=lambda item: (
            item.observations[0].timestamp_seconds if item.observations else 0.0,
            item.track_id,
            item.appearance_index,
        ),
    )


def build_single_track_appearances(
    frame_records: list[TrackedFrameRecord],
) -> list[TrackAppearance]:
    ordered_records = sorted(
        frame_records,
        key=lambda item: item.timestamp_seconds,
    )
    active_indexes_by_track: dict[str, list[int]] = {}
    present_indexes_by_track: dict[str, list[int]] = {}

    for frame_index, record in enumerate(ordered_records):
        seen_track_ids: set[str] = set()
        for track in record.tracks:
            seen_track_ids.add(track.track_id)
            if track.state == "active":
                active_indexes_by_track.setdefault(track.track_id, []).append(
                    frame_index
                )
        for track_id in seen_track_ids:
            present_indexes_by_track.setdefault(track_id, []).append(frame_index)

    appearances: list[TrackAppearance] = []
    for track_id, active_indexes in active_indexes_by_track.items():
        if not active_indexes:
            continue
        start_index = min(active_indexes)
        end_index = max(
            max(active_indexes),
            max(present_indexes_by_track.get(track_id, active_indexes)),
        )
        observations: list[TrackedFrameObservation] = []
        for record in ordered_records[start_index : end_index + 1]:
            observations.append(
                TrackedFrameObservation(
                    image_path=record.image_path,
                    timestamp_seconds=record.timestamp_seconds,
                    track=active_track_by_id(record.tracks, track_id),
                )
            )
        appearances.append(
            TrackAppearance(
                track_id=track_id,
                appearance_index=1,
                observations=observations,
            )
        )

    return sorted(
        appearances,
        key=lambda item: (
            item.observations[0].timestamp_seconds if item.observations else 0.0,
            item.track_id,
        ),
    )


def active_track_by_id(tracks: list[Track], track_id: str) -> Track | None:
    for track in tracks:
        if track.track_id == track_id and track.state == "active":
            return track
    return None


def write_tracked_object_clips(
    source_video_path: Path,
    frame_records: list[TrackedFrameRecord],
    output_dir: Path,
    output_fps: float,
    single_appearance: bool = False,
) -> list[dict]:
    appearances = (
        build_single_track_appearances(frame_records)
        if single_appearance
        else build_continuous_track_appearances(frame_records)
    )
    if not appearances:
        return []

    source_start = parse_source_video_start_datetime(source_video_path)
    clips_dir = output_dir / "tracked_object_clips"
    clip_results: list[dict] = []
    for appearance in appearances:
        output_path = tracked_object_clip_output_path(
            source_video_path,
            clips_dir,
            appearance.track_id,
            appearance.appearance_index,
        )
        with tempfile.TemporaryDirectory(prefix="tracked_object_clip_frames_") as temp_dir:
            annotated_frames: list[Path] = []
            temp_path = Path(temp_dir)
            for frame_index, observation in enumerate(appearance.observations):
                annotated_path = temp_path / f"frame_{frame_index:06d}.jpg"
                draw_tracked_object_clip_frame(
                    observation.image_path,
                    annotated_path,
                    observation.track,
                    timestamp_text=format_wall_clock_timestamp(
                        source_start,
                        observation.timestamp_seconds,
                    ),
                )
                annotated_frames.append(annotated_path)
            write_image_sequence_video(
                annotated_frames,
                output_path,
                fps=output_fps,
            )

        first_observation = appearance.observations[0]
        last_observation = appearance.observations[-1]
        clip_results.append(
            {
                "track_id": appearance.track_id,
                "appearance_index": appearance.appearance_index,
                "frame_count": len(appearance.observations),
                "visible_frame_count": sum(
                    1 for observation in appearance.observations if observation.track
                ),
                "start_seconds": first_observation.timestamp_seconds,
                "end_seconds": last_observation.timestamp_seconds,
                "start_timestamp": format_wall_clock_timestamp(
                    source_start,
                    first_observation.timestamp_seconds,
                ),
                "end_timestamp": format_wall_clock_timestamp(
                    source_start,
                    last_observation.timestamp_seconds,
                ),
                "video_path": str(output_path),
            }
        )
    return clip_results


def sam3_tracked_clip_frame_record(
    source_frame: SampledFrame,
    frame_result: Sam3FrameResult,
    full_frame_tracks: list[Track],
    roi: ROI | None,
) -> TrackedFrameRecord:
    if roi is None:
        return TrackedFrameRecord(
            image_path=source_frame.path,
            timestamp_seconds=frame_result.timestamp_seconds,
            tracks=full_frame_tracks,
        )
    return TrackedFrameRecord(
        image_path=frame_result.image_path,
        timestamp_seconds=frame_result.timestamp_seconds,
        tracks=frame_result.tracks,
    )


def tracked_object_clip_output_path(
    source_video_path: Path,
    clips_dir: Path,
    track_id: str,
    appearance_index: int,
) -> Path:
    safe_track_id = safe_path_component(track_id)
    return clips_dir / (
        f"{source_video_path.stem}_track-{safe_track_id}"
        f"_appearance-{appearance_index}.mp4"
    )


def safe_path_component(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return normalized or "unknown"


def parse_source_video_start_datetime(video_path: Path) -> datetime | None:
    try:
        return datetime.strptime(video_path.stem, "%Y-%m-%d_%H-%M-%S")
    except ValueError:
        return None


def format_wall_clock_timestamp(
    source_start: datetime | None,
    timestamp_seconds: float,
) -> str:
    if source_start is None:
        return f"{timestamp_seconds:.2f}s"
    timestamp = source_start + timedelta(seconds=timestamp_seconds)
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def draw_tracked_object_clip_frame(
    image_path: Path,
    output_path: Path,
    track: Track | None,
    timestamp_text: str,
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "`--tracked-object-clips` requires OpenCV. Install it with "
            "`pip install opencv-python`."
        ) from exc

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image for clip annotation: {image_path}")

    draw_timestamp(image, timestamp_text)
    if track is not None and track.state == "active":
        x1, y1, x2, y2 = [int(round(value)) for value in track.bbox.as_xyxy()]
        label = (
            f"track-{track.track_id} "
            f"{track.label}-{track.confidence * 100:.2f}%"
        )
        color = color_for_track(track.track_id)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        draw_label(image, label, x1, y1, color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not write clip frame: {output_path}")


def draw_timestamp(image: object, text: str) -> None:
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    margin = 8
    x = 12
    y = 12
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_width, text_height = text_size
    top_left = (x, y)
    bottom_right = (
        x + text_width + margin * 2,
        y + text_height + baseline + margin * 2,
    )
    text_origin = (x + margin, y + margin + text_height)
    cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)
    cv2.putText(
        image,
        text,
        text_origin,
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def detect_input_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"

    supported = ", ".join(sorted(IMAGE_EXTENSIONS | VIDEO_EXTENSIONS))
    raise ValueError(
        f"Unsupported input type for {path}. Supported extensions: {supported}"
    )


def build_detector(model_config_path: str, classes: list[str] | None) -> RFDETRDetector:
    model_mapping = load_yaml(model_config_path)
    if classes is not None:
        model_mapping.setdefault("classes", {})["allow"] = classes
    return RFDETRDetector(RFDETRConfig.from_mapping(model_mapping))


def build_sam3_adapter(
    model_config_path: str,
    *,
    include_masks: bool = False,
) -> SAM3ModelAdapter:
    model_mapping = load_yaml(model_config_path)
    config = SAM3Config.from_mapping(model_mapping)
    if include_masks and not config.include_masks:
        config = replace(config, include_masks=True)
    return SAM3ModelAdapter(config)


def build_sam3_live_config(
    args: argparse.Namespace,
    use_case_mapping: dict,
) -> Sam3LiveConfig:
    live_mapping = dict(use_case_mapping.get("sam3_live", {}))
    input_mapping = use_case_mapping.get("input", {})
    if "sample_fps" not in live_mapping and "fps" in input_mapping:
        live_mapping["sample_fps"] = input_mapping["fps"]

    sam3_model_mapping = use_case_mapping.get("models", {}).get("sam3", {})
    if "prompt" not in live_mapping and "prompt" in sam3_model_mapping:
        live_mapping["prompt"] = sam3_model_mapping["prompt"]

    live_config = Sam3LiveConfig.from_mapping(live_mapping)
    overrides = {}
    if args.strategy is not None:
        overrides["strategy"] = args.strategy
    if args.sample_fps is not None:
        overrides["sample_fps"] = args.sample_fps
    if args.window_seconds is not None:
        overrides["window_seconds"] = args.window_seconds
    if args.stride_seconds is not None:
        overrides["stride_seconds"] = args.stride_seconds
    if args.dwell_threshold_seconds is not None:
        overrides["dwell_threshold_seconds"] = args.dwell_threshold_seconds
    if args.missing_grace_seconds is not None:
        overrides["missing_grace_seconds"] = args.missing_grace_seconds
    if args.match_iou_threshold is not None:
        overrides["match_iou_threshold"] = args.match_iou_threshold
    if args.match_distance_threshold_px is not None:
        overrides["match_distance_threshold_px"] = args.match_distance_threshold_px
    if args.prompt is not None:
        overrides["prompt"] = tuple(args.prompt)
    if args.roi is not None:
        overrides["roi"] = parse_roi(args.roi)
    return replace(live_config, **overrides)


def resolve_sam3_live_model_config(
    args: argparse.Namespace,
    use_case_mapping: dict,
    use_case_config_path: Path | None,
) -> Path:
    model_config = args.model_config
    if model_config is None:
        model_config = (
            use_case_mapping.get("models", {})
            .get("sam3", {})
            .get("config", "vision_pipeline/configs/models/sam3.yaml")
        )
    model_config_path = Path(model_config)
    if model_config_path.is_absolute() or use_case_config_path is None:
        return model_config_path
    return resolve_config_path(use_case_config_path, model_config_path)


def resolve_sam3_live_source(
    args: argparse.Namespace,
    use_case_mapping: dict,
    use_case_config_path: Path | None,
) -> tuple[str, str]:
    if args.input:
        return args.input, detect_live_source_type(args.input)

    input_mapping = use_case_mapping.get("input", {})
    source_mapping = input_mapping.get("source")
    if source_mapping is None and input_mapping.get("camera_config"):
        if use_case_config_path is None:
            raise ValueError("camera_config requires --config")
        camera_path = resolve_config_path(
            use_case_config_path,
            input_mapping["camera_config"],
        )
        camera_mapping = load_yaml(camera_path)
        source_mapping = camera_mapping.get("input", {})

    if not isinstance(source_mapping, dict):
        raise ValueError("sam3-live requires an input path/RTSP URL or config source")

    source_type = str(source_mapping.get("type", "file"))
    uri = str(source_mapping.get("uri") or "").strip()
    uri_env = str(source_mapping.get("uri_env") or "").strip()
    if not uri and uri_env:
        uri = os.environ.get(uri_env, "").strip()
    if not uri:
        raise ValueError(
            "No source URI found. Pass an input path/RTSP URL, set input.source.uri, "
            "or export the configured uri_env."
        )
    return uri, "rtsp" if source_type == "rtsp" else detect_live_source_type(uri)


def detect_live_source_type(value: str) -> str:
    text = value.strip().lower()
    if text.startswith(("rtsp://", "rtsps://")):
        return "rtsp"
    return "file"


def build_qwen3_vl_client(
    model_config_path: str,
    endpoint: str | None,
) -> Qwen3VLClient:
    model_mapping = load_yaml(model_config_path)
    return Qwen3VLClient(Qwen3VLConfig.from_mapping(model_mapping, endpoint=endpoint))


def run_qwen3_vl_on_frame(
    client: Qwen3VLClient,
    image_path: Path,
    timestamp_seconds: float,
    args: argparse.Namespace,
) -> dict:
    if args.task == "detect":
        target = args.prompt or client.config.default_target
        result = client.detect_image_path(
            image_path,
            target=target,
            timestamp_seconds=timestamp_seconds,
            bbox_coordinate_format=args.bbox_format,
        )
    else:
        prompt = args.prompt or (
            "Describe the image for restaurant operations. Return JSON with keys "
            "summary and observations."
        )
        result = client.analyze_image_path(
            image_path,
            prompt=prompt,
            timestamp_seconds=timestamp_seconds,
            response_format=client.config.response_format,
        )

    return {
        "prompt": result.prompt,
        "raw_result": result.raw_result,
        "detections": result.detections,
    }


def detection_classes(args: argparse.Namespace) -> list[str] | None:
    if args.classes is not None:
        return args.classes
    if args.tracker != "none":
        return args.track_classes
    return None


def build_tracker(args: argparse.Namespace) -> BoTSORTTracker | None:
    if args.tracker == "none":
        return None
    tracker_mapping = load_yaml(args.tracker_config)
    return BoTSORTTracker(BoTSORTConfig.from_mapping(tracker_mapping))


def trackable_detections(
    detections: list[Detection],
    args: argparse.Namespace,
) -> list[Detection]:
    allowed = {label.lower() for label in args.track_classes}
    return [
        detection
        for detection in detections
        if not allowed or detection.label.lower() in allowed
    ]


def serialize_detections(
    detections: list[Detection],
    *,
    include_contours: bool = False,
    contour_epsilon_px: float = 2.0,
) -> list[dict]:
    serialized = []
    for detection in detections:
        item = {
            "label": detection.label,
            "confidence": detection.confidence,
            "bbox_xyxy": detection.bbox.as_xyxy(),
            "metadata": serialize_model_metadata(detection.metadata),
        }
        if include_contours:
            item["contours_xy"] = detection_contours_xy(
                detection,
                epsilon_px=contour_epsilon_px,
            )
            item["contours_format"] = "absolute_xy"
        serialized.append(item)
    return serialized


def serialize_tracks(
    tracks: list[Track],
    *,
    include_contours: bool = False,
    contour_epsilon_px: float = 2.0,
) -> list[dict]:
    serialized = []
    for track in tracks:
        item = {
            "track_id": track.track_id,
            "label": track.label,
            "confidence": track.confidence,
            "state": track.state,
            "bbox_xyxy": track.bbox.as_xyxy(),
            "metadata": serialize_model_metadata(track.metadata),
        }
        if include_contours:
            item["contours_xy"] = track_contours_xy(
                track,
                epsilon_px=contour_epsilon_px,
            )
            item["contours_format"] = "absolute_xy"
        serialized.append(item)
    return serialized


def materialize_frame_result_contours(
    frame_results: list[Sam3FrameResult],
    *,
    include_contours: bool,
    contour_epsilon_px: float = 2.0,
) -> None:
    for frame_result in frame_results:
        for track in frame_result.tracks:
            if include_contours and "contours_xy" not in track.metadata:
                contours = track_contours_xy(
                    track,
                    epsilon_px=contour_epsilon_px,
                )
                if contours:
                    track.metadata["contours_xy"] = contours
                    track.metadata["contours_format"] = "absolute_xy"
            track.metadata.pop("mask", None)


def serialize_model_metadata(metadata: dict) -> dict:
    serialized = dict(metadata)
    mask = serialized.pop("mask", None)
    if mask is not None:
        mask_shape = getattr(mask, "shape", None)
        serialized["has_mask"] = True
        if mask_shape is not None:
            serialized["mask_shape"] = [int(value) for value in mask_shape]
    return serialized


def strip_track_masks(tracks: list[Track]) -> None:
    for track in tracks:
        track.metadata.pop("mask", None)


def detection_contours_xy(
    detection: Detection,
    *,
    epsilon_px: float = 2.0,
) -> list[list[list[int]]]:
    mask_offset, mask_canvas_size = mask_transform_from_metadata(detection.metadata)
    return contours_from_mask(
        detection.metadata.get("mask"),
        epsilon_px=epsilon_px,
        mask_offset=mask_offset,
        mask_canvas_size=mask_canvas_size,
    )


def track_contours_xy(
    track: Track,
    *,
    epsilon_px: float = 2.0,
) -> list[list[list[int]]]:
    mask_offset, mask_canvas_size = mask_transform_from_metadata(track.metadata)
    contours_xy = track.metadata.get("contours_xy")
    if contours_xy is not None:
        return normalize_contours_xy(contours_xy, offset=mask_offset)
    return contours_from_mask(
        track.metadata.get("mask"),
        epsilon_px=epsilon_px,
        mask_offset=mask_offset,
        mask_canvas_size=mask_canvas_size,
    )


def mask_transform_from_metadata(
    metadata: dict,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    roi = metadata.get("roi_xyxy")
    if not roi:
        return None, None
    x1, y1, x2, y2 = [int(round(value)) for value in roi]
    return (x1, y1), (x2 - x1, y2 - y1)


def normalize_contours_xy(
    contours_xy: object,
    *,
    offset: tuple[int, int] | None = None,
) -> list[list[list[int]]]:
    if not isinstance(contours_xy, list):
        return []
    x_offset, y_offset = offset if offset is not None else (0, 0)
    normalized: list[list[list[int]]] = []
    for contour in contours_xy:
        if not isinstance(contour, list):
            continue
        points: list[list[int]] = []
        for point in contour:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                x = int(round(float(point[0]))) + x_offset
                y = int(round(float(point[1]))) + y_offset
            except (TypeError, ValueError):
                continue
            points.append([x, y])
        if points:
            normalized.append(points)
    return normalized


def contours_from_mask(
    mask: object | None,
    *,
    epsilon_px: float = 2.0,
    mask_offset: tuple[int, int] | None = None,
    mask_canvas_size: tuple[int, int] | None = None,
) -> list[list[list[int]]]:
    if mask is None:
        return []

    try:
        import cv2
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "--contour-json requires OpenCV and NumPy. Install with "
            "`pip install opencv-python numpy`."
        ) from exc

    mask_u8 = mask_to_uint8(mask)
    if mask_u8 is None:
        return []

    if mask_canvas_size is not None:
        target_width, target_height = mask_canvas_size
        if mask_u8.shape[:2] != (target_height, target_width):
            mask_u8 = cv2.resize(
                mask_u8,
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST,
            )

    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return []

    if mask_offset is not None:
        x_offset, y_offset = mask_offset
        offset = np.array([[[x_offset, y_offset]]], dtype=contours[0].dtype)
        contours = [contour + offset for contour in contours]

    simplified_contours: list[list[list[int]]] = []
    epsilon = max(0.0, float(epsilon_px))
    for contour in contours:
        if epsilon > 0:
            contour = cv2.approxPolyDP(contour, epsilon, True)
        points = contour.reshape(-1, 2)
        simplified_contours.append(
            [[int(point[0]), int(point[1])] for point in points]
        )
    return simplified_contours


def annotated_variant_path(image_path: Path) -> Path:
    return image_path.with_name(f"{image_path.stem}_boxes{image_path.suffix}")


def output_image_path(image_path: Path, output_dir: str | None) -> Path:
    if output_dir:
        return Path(output_dir) / f"{image_path.stem}_boxes{image_path.suffix}"
    return annotated_variant_path(image_path)


def draw_detections(
    image_path: Path,
    output_path: Path,
    detections: list[Detection],
    *,
    draw_contours: bool = False,
    mask_offset: tuple[int, int] | None = None,
    mask_canvas_size: tuple[int, int] | None = None,
) -> None:
    """Draw detection boxes and labels onto an image."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "`--draw-boxes` requires OpenCV. Install it with `pip install opencv-python`."
        ) from exc

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image for annotation: {image_path}")

    for detection in detections:
        x1, y1, x2, y2 = [int(round(value)) for value in detection.bbox.as_xyxy()]
        label = f"{detection.label}-{detection.confidence * 100:.2f}%"
        color = (30, 220, 30)
        drew_contour = False
        if draw_contours:
            drew_contour = draw_mask_contour(
                image,
                detection.metadata.get("mask"),
                color,
                mask_offset=mask_offset,
                mask_canvas_size=mask_canvas_size,
            )
        if not drew_contour:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        draw_label(image, label, x1, y1, color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not write annotated image: {output_path}")


def draw_tracks(
    image_path: Path,
    output_path: Path,
    tracks: list[Track],
    *,
    draw_contours: bool = False,
    mask_offset: tuple[int, int] | None = None,
    mask_canvas_size: tuple[int, int] | None = None,
) -> None:
    """Draw track boxes and track IDs onto an image."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "`--draw-boxes` requires OpenCV. Install it with `pip install opencv-python`."
        ) from exc

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read image for annotation: {image_path}")

    for track in tracks:
        if track.state != "active":
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in track.bbox.as_xyxy()]
        label = (
            f"track-{track.track_id} "
            f"{track.label}-{track.confidence * 100:.2f}%"
        )
        color = color_for_track(track.track_id)
        drew_contour = False
        if draw_contours:
            drew_contour = draw_track_contour(
                image,
                track,
                color,
                mask_offset=mask_offset,
                mask_canvas_size=mask_canvas_size,
            )
        if not drew_contour:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        draw_label(image, label, x1, y1, color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Could not write annotated image: {output_path}")


def draw_tracked_video(
    source_video_path: Path,
    output_path: Path,
    frame_results: list[Sam3FrameResult],
    *,
    draw_boxes: bool,
    draw_contours: bool,
    roi: ROI | None,
    output_fps: float | None = None,
    frame_result_fps: float | None = None,
) -> None:
    """Draw SAM3 tracks onto a source video and write an MP4."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Writing annotated SAM3 videos requires OpenCV. Install it with "
            "`pip install opencv-python`."
        ) from exc

    capture = cv2.VideoCapture(str(source_video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for annotation: {source_video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    target_fps = draw_video_output_fps(fps, output_fps)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target_frame_indexes = sampled_draw_video_frame_indexes(
        total_frames,
        source_fps=fps,
        output_fps=target_fps,
    )
    write_all_frames = target_frame_indexes is None
    output_total_frames = (
        total_frames if write_all_frames else len(target_frame_indexes)
    )
    progress_interval = max(1, int(round(target_fps * 10)))
    total_text = str(output_total_frames) if output_total_frames > 0 else "?"
    print(
        f"draw_video: writing {output_path} "
        f"frames={total_text} fps={target_fps:.2f} "
        f"source_fps={fps:.2f} "
        f"preset={ANNOTATION_VIDEO_PRESET} crf={ANNOTATION_VIDEO_CRF}",
        flush=True,
    )
    try:
        writer = open_ffmpeg_video_writer(
            output_path,
            fps=target_fps,
            width=width,
            height=height,
        )
    except Exception:
        capture.release()
        raise

    results_by_frame = frame_results_by_video_frame(
        frame_results,
        video_fps=fps,
        frame_result_fps=frame_result_fps,
    )
    mask_offset = (roi[0], roi[1]) if roi else None
    mask_canvas_size = ((roi[2] - roi[0], roi[3] - roi[1]) if roi else None)
    frame_index = 0
    output_frame_count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if not write_all_frames and frame_index not in target_frame_indexes:
                frame_index += 1
                continue
            frame_result = results_by_frame.get(frame_index)
            if frame_result is not None:
                tracks = (
                    offset_tracks(frame_result.tracks, roi)
                    if roi
                    else frame_result.tracks
                )
                draw_track_annotations_on_frame(
                    frame,
                    tracks,
                    draw_boxes=draw_boxes,
                    draw_contours=draw_contours,
                    mask_offset=mask_offset,
                    mask_canvas_size=mask_canvas_size,
                )
            write_ffmpeg_video_frame(writer, frame, output_path)
            output_frame_count += 1
            frame_index += 1
            if output_frame_count % progress_interval == 0:
                print(
                    f"draw_video: frame={output_frame_count}/{total_text}",
                    flush=True,
                )
    finally:
        capture.release()
        close_ffmpeg_video_writer(writer, output_path)
    print(
        f"draw_video: done frames={output_frame_count} "
        f"source_frames={frame_index}",
        flush=True,
    )


def draw_video_output_fps(source_fps: float, requested_fps: float | None) -> float:
    if requested_fps is None or requested_fps <= 0:
        return source_fps
    return min(source_fps, requested_fps)


def frame_results_by_video_frame(
    frame_results: list[Sam3FrameResult],
    *,
    video_fps: float,
    frame_result_fps: float | None = None,
) -> dict[int, Sam3FrameResult]:
    """Index results against the exact video that will be rendered."""
    use_timestamps = (
        frame_result_fps is not None
        and not math.isclose(
            frame_result_fps,
            video_fps,
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
    )
    indexed = {}
    for frame_result in frame_results:
        if use_timestamps:
            frame_index = seconds_to_frame_index(
                frame_result.timestamp_seconds,
                video_fps,
            )
        else:
            frame_index = frame_result.frame_index
        if frame_index is not None:
            indexed[frame_index] = frame_result
    return indexed


def sampled_draw_video_frame_indexes(
    total_frames: int,
    *,
    source_fps: float,
    output_fps: float,
) -> set[int] | None:
    if total_frames <= 0 or output_fps >= source_fps - 1e-6:
        return None

    duration_seconds = total_frames / source_fps
    output_frame_count = max(1, int(math.ceil(duration_seconds * output_fps - 1e-9)))
    frame_indexes: set[int] = set()
    last_frame_index = -1
    for output_index in range(output_frame_count):
        timestamp_seconds = output_index / output_fps
        frame_index = min(
            total_frames - 1,
            seconds_to_frame_index(timestamp_seconds, source_fps),
        )
        if frame_index <= last_frame_index:
            frame_index = min(total_frames - 1, last_frame_index + 1)
        if frame_index >= total_frames:
            break
        frame_indexes.add(frame_index)
        last_frame_index = frame_index
    return frame_indexes


def draw_serialized_contour_video(
    source_video_path: Path,
    contour_json_path: Path,
    output_path: Path,
    *,
    draw_boxes: bool = False,
    draw_labels: bool = True,
    contour_thickness: int = 2,
    box_thickness: int = 2,
    crf: int = ANNOTATION_VIDEO_CRF,
    preset: str = ANNOTATION_VIDEO_PRESET,
) -> dict[str, int]:
    """Draw saved contour JSON annotations onto a source video."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Drawing contour videos requires OpenCV. Install it with "
            "`pip install opencv-python`."
        ) from exc

    contour_payload = load_contour_json(contour_json_path)
    capture = cv2.VideoCapture(str(source_video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video for annotation: {source_video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    annotations_by_frame = serialized_contour_annotations_by_frame(
        contour_payload,
        fps=fps,
    )

    try:
        writer = open_ffmpeg_video_writer(
            output_path,
            fps=fps,
            width=width,
            height=height,
            crf=crf,
            preset=preset,
        )
    except Exception:
        capture.release()
        raise

    frame_count = 0
    annotated_frames = 0
    drawn_annotations = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            annotations = annotations_by_frame.get(frame_count, [])
            if annotations:
                drawn_count = draw_serialized_annotations_on_frame(
                    frame,
                    annotations,
                    draw_boxes=draw_boxes,
                    draw_labels=draw_labels,
                    contour_thickness=contour_thickness,
                    box_thickness=box_thickness,
                )
                if drawn_count:
                    annotated_frames += 1
                    drawn_annotations += drawn_count
            write_ffmpeg_video_frame(writer, frame, output_path)
            frame_count += 1
    finally:
        capture.release()
        close_ffmpeg_video_writer(writer, output_path)

    return {
        "frames_read": frame_count,
        "annotated_frames": annotated_frames,
        "drawn_annotations": drawn_annotations,
    }


def open_ffmpeg_video_writer(
    output_path: Path,
    *,
    fps: float,
    width: int,
    height: int,
    crf: int = ANNOTATION_VIDEO_CRF,
    preset: str = ANNOTATION_VIDEO_PRESET,
) -> subprocess.Popen:
    """Open an ffmpeg/libx264 process that accepts raw OpenCV BGR frames."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError(
            "Writing annotated videos requires ffmpeg. Install it with "
            "`sudo apt install -y ffmpeg`."
        )
    if width <= 0 or height <= 0:
        raise RuntimeError(f"Invalid video writer size: {width}x{height}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        f"{fps:.6f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    try:
        return subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except OSError as exc:
        raise RuntimeError(
            f"Could not open ffmpeg video writer: {output_path}"
        ) from exc


def write_ffmpeg_video_frame(
    writer: subprocess.Popen,
    frame: object,
    output_path: Path,
) -> None:
    if writer.stdin is None:
        raise RuntimeError(f"ffmpeg video writer stdin is closed: {output_path}")
    try:
        writer.stdin.write(frame.tobytes())
    except BrokenPipeError as exc:
        stderr = read_process_stderr(writer)
        raise RuntimeError(
            f"ffmpeg video writer stopped while writing {output_path}: {stderr}"
        ) from exc


def close_ffmpeg_video_writer(writer: subprocess.Popen, output_path: Path) -> None:
    if writer.stdin is not None and not writer.stdin.closed:
        writer.stdin.close()
    returncode = writer.wait()
    stderr = read_process_stderr(writer)
    if returncode != 0:
        raise RuntimeError(f"ffmpeg failed while writing {output_path}: {stderr}")


def read_process_stderr(process: subprocess.Popen) -> str:
    if process.stderr is None:
        return ""
    try:
        return process.stderr.read().decode("utf-8", errors="replace").strip()
    except Exception:
        return ""


def load_contour_json(contour_json_path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(contour_json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid contour JSON: {contour_json_path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Contour JSON must be an object with a frames list.")
    return payload


def normalized_contour_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if isinstance(payload.get("frames"), list):
        return payload
    result = payload.get("result")
    if isinstance(result, dict) and isinstance(result.get("frames"), list):
        return result
    raise RuntimeError("Contour JSON must contain a frames list.")


def serialized_contour_annotations_by_frame(
    payload: dict[str, Any],
    *,
    fps: float,
) -> dict[int, list[dict[str, Any]]]:
    payload = normalized_contour_payload(payload)
    frames = payload["frames"]
    annotations_by_frame: dict[int, list[dict[str, Any]]] = {}
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        frame_index = serialized_contour_frame_index(frame, fps=fps)
        if frame_index is None:
            continue
        annotations = serialized_frame_annotations(frame)
        if annotations:
            annotations_by_frame.setdefault(frame_index, []).extend(annotations)
    return annotations_by_frame


def serialized_contour_frame_index(
    frame: dict[str, Any],
    *,
    fps: float,
) -> int | None:
    raw_frame_index = frame.get("frame_index")
    if raw_frame_index is not None:
        try:
            return max(0, int(raw_frame_index))
        except (TypeError, ValueError):
            return None

    timestamp_seconds = frame.get("timestamp_seconds")
    if timestamp_seconds is None:
        return None
    try:
        return max(0, int(round(float(timestamp_seconds) * fps)))
    except (TypeError, ValueError):
        return None


def serialized_frame_annotations(frame: dict[str, Any]) -> list[dict[str, Any]]:
    annotations = []
    for key in ("tracks", "detections"):
        values = frame.get(key)
        if isinstance(values, list):
            annotations.extend(value for value in values if isinstance(value, dict))
    return annotations


def draw_serialized_annotations_on_frame(
    image: object,
    annotations: list[dict[str, Any]],
    *,
    draw_boxes: bool,
    draw_labels: bool,
    contour_thickness: int,
    box_thickness: int,
) -> int:
    import cv2

    drawn_count = 0
    for annotation in annotations:
        if annotation.get("state", "active") != "active":
            continue

        annotation_id = serialized_annotation_id(annotation)
        color = color_for_track(annotation_id)
        drew_contour, contour_anchor = draw_serialized_contours(
            image,
            annotation.get("contours_xy"),
            color,
            thickness=contour_thickness,
        )
        bbox = serialized_bbox_xyxy(annotation)
        drew_box = False
        if draw_boxes and bbox is not None:
            x1, y1, x2, y2 = [int(round(value)) for value in bbox]
            cv2.rectangle(image, (x1, y1), (x2, y2), color, max(1, box_thickness))
            drew_box = True

        if not drew_contour and not drew_box:
            continue

        if draw_labels:
            anchor = contour_anchor
            if anchor is None and bbox is not None:
                anchor = (int(round(bbox[0])), int(round(bbox[1])))
            if anchor is not None:
                draw_label(
                    image,
                    serialized_annotation_label(annotation),
                    anchor[0],
                    anchor[1],
                    color,
                )
        drawn_count += 1
    return drawn_count


def serialized_annotation_id(annotation: dict[str, Any]) -> str:
    track_id = annotation.get("track_id")
    if track_id is not None:
        return str(track_id)
    label = annotation.get("label")
    if label is not None:
        return str(label)
    return "object"


def serialized_annotation_label(annotation: dict[str, Any]) -> str:
    label = str(annotation.get("label") or "object")
    track_id = annotation.get("track_id")
    prefix = f"track-{track_id} " if track_id is not None else ""
    confidence = annotation.get("confidence")
    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
        return f"{prefix}{label}-{confidence * 100:.2f}%"
    return f"{prefix}{label}"


def serialized_bbox_xyxy(
    annotation: dict[str, Any],
) -> tuple[float, float, float, float] | None:
    raw_bbox = annotation.get("bbox_xyxy")
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in raw_bbox]
    except (TypeError, ValueError):
        return None
    return (x1, y1, x2, y2)


def draw_serialized_contours(
    image: object,
    contours_xy: object,
    color: tuple[int, int, int],
    *,
    thickness: int,
) -> tuple[bool, tuple[int, int] | None]:
    import cv2
    import numpy as np

    if not isinstance(contours_xy, list):
        return False, None

    cv_contours = []
    all_points: list[tuple[int, int]] = []
    for contour in contours_xy:
        if not isinstance(contour, list):
            continue
        points = []
        for point in contour:
            if not isinstance(point, (list, tuple)) or len(point) < 2:
                continue
            try:
                x = int(round(float(point[0])))
                y = int(round(float(point[1])))
            except (TypeError, ValueError):
                continue
            points.append([x, y])
            all_points.append((x, y))
        if len(points) >= 2:
            cv_contours.append(np.asarray(points, dtype=np.int32).reshape((-1, 1, 2)))

    if not cv_contours:
        return False, None

    cv2.drawContours(image, cv_contours, -1, color, max(1, thickness))
    min_x = min(point[0] for point in all_points)
    min_y = min(point[1] for point in all_points)
    return True, (min_x, min_y)


def draw_track_annotations_on_frame(
    image: object,
    tracks: list[Track],
    *,
    draw_boxes: bool,
    draw_contours: bool,
    mask_offset: tuple[int, int] | None = None,
    mask_canvas_size: tuple[int, int] | None = None,
) -> None:
    import cv2

    for track in tracks:
        if track.state != "active":
            continue
        x1, y1, x2, y2 = [int(round(value)) for value in track.bbox.as_xyxy()]
        label = f"track-{track.track_id} {track.label}-{track.confidence * 100:.2f}%"
        color = color_for_track(track.track_id)
        drew_contour = False
        if draw_contours:
            drew_contour = draw_track_contour(
                image,
                track,
                color,
                mask_offset=mask_offset,
                mask_canvas_size=mask_canvas_size,
            )
        if draw_boxes or not drew_contour:
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        draw_label(image, label, x1, y1, color)


def draw_track_contour(
    image: object,
    track: Track,
    color: tuple[int, int, int],
    *,
    mask_offset: tuple[int, int] | None = None,
    mask_canvas_size: tuple[int, int] | None = None,
) -> bool:
    if track.metadata.get("contours_xy") is not None:
        drew_contour, _ = draw_serialized_contours(
            image,
            track_contours_xy(track, epsilon_px=0),
            color,
            thickness=2,
        )
        return drew_contour
    return draw_mask_contour(
        image,
        track.metadata.get("mask"),
        color,
        mask_offset=mask_offset,
        mask_canvas_size=mask_canvas_size,
    )


def draw_mask_contour(
    image: object,
    mask: object | None,
    color: tuple[int, int, int],
    *,
    mask_offset: tuple[int, int] | None = None,
    mask_canvas_size: tuple[int, int] | None = None,
) -> bool:
    import cv2
    import numpy as np

    if mask is None:
        return False
    mask_u8 = mask_to_uint8(mask)
    if mask_u8 is None:
        return False

    image_height, image_width = image.shape[:2]
    target_width, target_height = (
        mask_canvas_size
        if mask_canvas_size is not None
        else (image_width, image_height)
    )
    if mask_u8.shape[:2] != (target_height, target_width):
        mask_u8 = cv2.resize(
            mask_u8,
            (target_width, target_height),
            interpolation=cv2.INTER_NEAREST,
        )

    contours, _ = cv2.findContours(
        mask_u8,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if not contours:
        return False
    if mask_offset is not None:
        x_offset, y_offset = mask_offset
        offset = np.array([[[x_offset, y_offset]]], dtype=contours[0].dtype)
        contours = [contour + offset for contour in contours]
    cv2.drawContours(image, contours, -1, color, 2)
    return True


def mask_to_uint8(mask: object) -> object | None:
    try:
        import numpy as np
    except ImportError:
        return None

    if hasattr(mask, "detach"):
        mask = mask.detach().cpu()
    if hasattr(mask, "numpy"):
        array = mask.numpy()
    else:
        array = np.asarray(mask)

    array = np.squeeze(array)
    if array.ndim != 2:
        return None
    if array.dtype == np.bool_:
        return array.astype("uint8") * 255
    return (array > 0).astype("uint8") * 255


def color_for_track(track_id: str) -> tuple[int, int, int]:
    seed = sum(ord(char) for char in track_id)
    return (
        40 + (seed * 37) % 180,
        40 + (seed * 53) % 180,
        40 + (seed * 97) % 180,
    )


def draw_label(image: object, label: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    """Draw a filled label background plus text."""
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    margin = 4
    text_size, baseline = cv2.getTextSize(label, font, font_scale, thickness)
    text_width, text_height = text_size
    label_y = max(y, text_height + baseline + margin * 2)
    top_left = (x, label_y - text_height - baseline - margin * 2)
    bottom_right = (x + text_width + margin * 2, label_y)

    cv2.rectangle(image, top_left, bottom_right, color, -1)
    cv2.putText(
        image,
        label,
        (x + margin, label_y - baseline - margin),
        font,
        font_scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


if __name__ == "__main__":
    raise SystemExit(main())
