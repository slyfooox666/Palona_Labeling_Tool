"""Video metadata and nearest-frame extraction using PyAV."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import av
import cv2

from palona_depth.models import ControlFrame, ExtractedFrame, VideoMetadata


class VideoDataError(ValueError):
    """Raised when the source video cannot be decoded or aligned."""


def probe_video(path: Path) -> VideoMetadata:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise VideoDataError(f"Video does not exist: {path}")
    try:
        with av.open(str(path)) as container:
            stream = next((item for item in container.streams if item.type == "video"), None)
            if stream is None:
                raise VideoDataError(f"No video stream found in {path}")
            rate = stream.average_rate or stream.guessed_rate
            fps = float(rate) if rate is not None else 0.0
            duration = (
                float(stream.duration * stream.time_base)
                if stream.duration is not None and stream.time_base is not None
                else float(container.duration / av.time_base) if container.duration is not None else 0.0
            )
            frame_count = int(stream.frames) if stream.frames else None
            if fps <= 0 or not math.isfinite(fps):
                raise VideoDataError(f"Invalid video FPS for {path}: {fps}")
            return VideoMetadata(
                width=int(stream.codec_context.width),
                height=int(stream.codec_context.height),
                fps=fps,
                duration_seconds=max(0.0, duration),
                frame_count=frame_count,
            )
    except (av.error.FFmpegError, OSError) as exc:
        raise VideoDataError(f"Could not open video {path}: {exc}") from exc


def extract_aligned_frames(
    video_path: Path,
    control_frames: Iterable[ControlFrame],
    output_dir: Path,
    *,
    tolerance_seconds: float,
) -> list[ExtractedFrame]:
    targets = list(control_frames)
    if not targets:
        raise VideoDataError("No Control frames were selected")
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[ExtractedFrame] = []
    target_index = 0
    previous = None
    previous_time = 0.0
    previous_index = 0
    origin: float | None = None

    try:
        with av.open(str(video_path)) as container:
            stream = next((item for item in container.streams if item.type == "video"), None)
            if stream is None:
                raise VideoDataError(f"No video stream found in {video_path}")
            stream.thread_type = "AUTO"
            for decoded_index, frame in enumerate(container.decode(stream)):
                raw_time = float(frame.time) if frame.time is not None else decoded_index / float(stream.average_rate)
                if origin is None:
                    origin = raw_time
                timestamp = raw_time - origin
                while target_index < len(targets) and timestamp >= targets[target_index].timestamp_seconds:
                    target = targets[target_index]
                    choices = [(frame, timestamp, decoded_index)]
                    if previous is not None:
                        choices.append((previous, previous_time, previous_index))
                    chosen, chosen_time, chosen_index = min(
                        choices,
                        key=lambda item: abs(item[1] - target.timestamp_seconds),
                    )
                    alignment_error = abs(chosen_time - target.timestamp_seconds)
                    if alignment_error > tolerance_seconds:
                        raise VideoDataError(
                            f"Frame {target.frame_index} depth/video alignment error "
                            f"{alignment_error:.6f}s exceeds {tolerance_seconds:.6f}s"
                        )
                    image_path = output_dir / f"frame_{target.frame_index:08d}.png"
                    bgr = chosen.to_ndarray(format="bgr24")
                    if not cv2.imwrite(str(image_path), bgr):
                        raise VideoDataError(f"Could not write extracted frame {image_path}")
                    results.append(
                        ExtractedFrame(
                            control=target,
                            image_path=image_path,
                            decoded_frame_index=chosen_index,
                            decoded_timestamp_seconds=chosen_time,
                            alignment_error_seconds=alignment_error,
                        )
                    )
                    target_index += 1
                if target_index >= len(targets):
                    break
                previous = frame
                previous_time = timestamp
                previous_index = decoded_index
    except (av.error.FFmpegError, OSError) as exc:
        raise VideoDataError(f"Could not decode video {video_path}: {exc}") from exc

    if len(results) != len(targets):
        missing = targets[len(results)]
        raise VideoDataError(
            f"Video ended before Control frame {missing.frame_index} at {missing.timestamp_seconds:.3f}s"
        )
    return results
