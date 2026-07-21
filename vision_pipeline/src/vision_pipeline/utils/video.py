"""Video helpers."""

from __future__ import annotations

import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path


def seconds_per_frame(fps: float) -> float:
    """Return frame interval for a configured FPS."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    return 1.0 / fps


@dataclass(frozen=True)
class VideoMetadata:
    """Basic video metadata from ffprobe."""

    path: Path
    width: int
    height: int
    duration_seconds: float
    fps: float


@dataclass(frozen=True)
class SampledFrame:
    """Extracted frame path with source timestamp."""

    path: Path
    timestamp_seconds: float


def probe_video(video_path: str | Path) -> VideoMetadata:
    """Probe an MP4/stream file with ffprobe."""
    ffprobe = _require_binary("ffprobe")
    path = Path(video_path)
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,duration:format=duration",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return video_metadata_from_ffprobe(path, json.loads(completed.stdout))


def video_metadata_from_ffprobe(path: Path, payload: dict) -> VideoMetadata:
    """Build VideoMetadata from ffprobe JSON output."""
    stream = payload["streams"][0]
    duration = _parse_optional_float(stream.get("duration"))
    if duration is None:
        duration = _parse_optional_float((payload.get("format") or {}).get("duration"))
    fps = _parse_ratio(stream.get("avg_frame_rate", "0/1"))
    if fps <= 0:
        fps = _parse_ratio(stream.get("r_frame_rate", "0/1"))
    return VideoMetadata(
        path=path,
        width=int(stream["width"]),
        height=int(stream["height"]),
        duration_seconds=duration or 0.0,
        fps=fps,
    )


def extract_sample_frames(
    video_path: str | Path,
    output_dir: str | Path,
    sample_count: int,
    image_ext: str = "jpg",
    start_from_seconds: float = 0.0,
    worker_threads: int = 1,
) -> list[SampledFrame]:
    """Extract evenly spaced frames from a video using ffmpeg."""
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    start_from_seconds = _validate_start_from(start_from_seconds)

    metadata = probe_video(video_path)
    if metadata.duration_seconds <= 0:
        timestamps = [start_from_seconds]
    elif start_from_seconds >= metadata.duration_seconds:
        timestamps = []
    else:
        sample_duration = metadata.duration_seconds - start_from_seconds
        step = sample_duration / (sample_count + 1)
        timestamps = [
            start_from_seconds + step * (index + 1)
            for index in range(sample_count)
        ]
    return extract_frames_at_timestamps(
        metadata.path,
        output_dir,
        timestamps,
        image_ext=image_ext,
        worker_threads=worker_threads,
    )


def extract_frames_at_fps(
    video_path: str | Path,
    output_dir: str | Path,
    sample_fps: float,
    max_frames: int | None = None,
    image_ext: str = "jpg",
    start_from_seconds: float = 0.0,
    worker_threads: int = 1,
) -> list[SampledFrame]:
    """Extract frames at a fixed FPS from a video using ffmpeg."""
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    start_from_seconds = _validate_start_from(start_from_seconds)

    metadata = probe_video(video_path)
    interval = 1.0 / sample_fps
    timestamps: list[float] = []
    timestamp = start_from_seconds
    while metadata.duration_seconds <= 0 or timestamp < metadata.duration_seconds:
        timestamps.append(timestamp)
        if max_frames is not None and len(timestamps) >= max_frames:
            break
        timestamp += interval
        if metadata.duration_seconds <= 0:
            break

    return extract_frames_at_timestamps(
        metadata.path,
        output_dir,
        timestamps,
        image_ext=image_ext,
        worker_threads=worker_threads,
    )


def extract_frames_at_timestamps(
    video_path: str | Path,
    output_dir: str | Path,
    timestamps: list[float],
    image_ext: str = "jpg",
    worker_threads: int = 1,
) -> list[SampledFrame]:
    """Extract frames at explicit timestamps."""
    ffmpeg = _require_binary("ffmpeg")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if worker_threads <= 0:
        raise ValueError("worker_threads must be positive")

    def extract_one(index_timestamp: tuple[int, float]) -> SampledFrame:
        index, timestamp = index_timestamp
        frame_path = output / f"frame_{index:04d}_{timestamp:.2f}s.{image_ext}"
        command = [
            ffmpeg,
            "-y",
            "-ss",
            f"{timestamp:.3f}",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(frame_path),
        ]
        subprocess.run(command, check=True, capture_output=True, text=True)
        return SampledFrame(path=frame_path, timestamp_seconds=timestamp)

    indexed_timestamps = list(enumerate(timestamps))
    if worker_threads == 1 or len(indexed_timestamps) <= 1:
        return [extract_one(item) for item in indexed_timestamps]

    max_workers = min(worker_threads, len(indexed_timestamps))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(extract_one, indexed_timestamps))


def _validate_start_from(value: float) -> float:
    start_from_seconds = float(value)
    if start_from_seconds < 0:
        raise ValueError("start_from_seconds must be non-negative")
    return start_from_seconds


def _parse_ratio(value: str) -> float:
    try:
        numerator_text, denominator_text = value.split("/")
        denominator = float(denominator_text)
        if denominator == 0:
            return 0.0
        return float(numerator_text) / denominator
    except (AttributeError, ValueError):
        return 0.0


def _parse_optional_float(value: object) -> float | None:
    if value in (None, "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if path:
        return path
    raise RuntimeError(
        f"Required command `{name}` was not found on PATH. Install the system "
        "ffmpeg package first, for example: `sudo apt install -y ffmpeg`."
    )
