"""Atomic, browser-compatible video transcoding through the pinned PyAV runtime.

The labeling UI matches a converted playback copy to its Control JSON by the
source stem.  This exporter therefore requires ``scene.mkv -> scene.mp4`` and
keeps the decoded frame order, nominal FPS, dimensions, and clip duration.  It
intentionally writes video only; audio preservation is not part of the MVP.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

import av


Progress = Callable[[str], None]


class VideoConvertError(ValueError):
    """Raised when a safe, alignment-preserving conversion cannot be made."""


@dataclass(frozen=True)
class VideoConvertOptions:
    input_path: Path
    output_path: Path
    codec: str = "libx264"
    crf: int = 18
    preset: str = "medium"


def convert_video(
    options: VideoConvertOptions,
    *,
    progress: Progress | None = None,
) -> dict[str, Any]:
    """Transcode one constant-rate video to H.264/yuv420p and install atomically."""

    report = progress or (lambda _message: None)
    source, target = _validated_paths(options.input_path, options.output_path)
    if not 0 <= options.crf <= 51:
        raise VideoConvertError("CRF must be an integer between 0 and 51")
    if not options.codec.strip():
        raise VideoConvertError("Video codec must not be empty")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = _temporary_mp4(target)
    try:
        report(f"Converting {source.name} to browser-compatible H.264 MP4")
        result = _write_video(
            source,
            temporary,
            codec=options.codec,
            crf=options.crf,
            preset=options.preset,
            progress=report,
        )
        _validate_encoded_video(temporary, expected=result)
        _fsync_file(temporary)
        os.replace(temporary, target)
        _fsync_directory(target.parent)
    except VideoConvertError:
        raise
    except (av.error.FFmpegError, OSError, ValueError) as exc:
        raise VideoConvertError(f"Could not convert {source}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)

    report("Video conversion complete")
    return {
        "input": str(source),
        "output": str(target),
        **result,
        "audio_preserved": False,
    }


def _write_video(
    source_path: Path,
    output_path: Path,
    *,
    codec: str,
    crf: int,
    preset: str,
    progress: Progress,
) -> dict[str, Any]:
    frame_count = 0
    with av.open(str(source_path), mode="r") as source:
        input_stream = next((stream for stream in source.streams if stream.type == "video"), None)
        if input_stream is None:
            raise VideoConvertError(f"No video stream found in {source_path}")

        rate = _stream_rate(input_stream, source_path)
        width = int(input_stream.codec_context.width)
        height = int(input_stream.codec_context.height)
        if width <= 0 or height <= 0:
            raise VideoConvertError(f"Source video has invalid dimensions: {width}x{height}")
        if width % 2 or height % 2:
            raise VideoConvertError(
                "H.264 yuv420p requires even source dimensions; refusing to resize "
                f"{width}x{height} because that would misalign contours"
            )

        source_duration = _duration_seconds(source, input_stream)
        input_stream.thread_type = "AUTO"
        frame_time_base = Fraction(rate.denominator, rate.numerator)

        with av.open(
            str(output_path),
            mode="w",
            format="mp4",
            options={"movflags": "faststart"},
        ) as destination:
            try:
                output_stream = destination.add_stream(codec, rate=rate)
            except (av.error.FFmpegError, ValueError) as exc:
                raise VideoConvertError(f"Video encoder {codec!r} is unavailable: {exc}") from exc
            output_stream.width = width
            output_stream.height = height
            output_stream.pix_fmt = "yuv420p"
            if codec in {"libx264", "h264"}:
                output_stream.options = {"crf": str(crf), "preset": preset}

            for frame_count, decoded in enumerate(source.decode(input_stream), start=1):
                # A sequential timestamp makes the browser copy explicitly CFR.
                # This preserves the frame-index/FPS mapping used by Control JSON.
                converted = decoded.reformat(width=width, height=height, format="yuv420p")
                converted.pts = frame_count - 1
                converted.time_base = frame_time_base
                for packet in output_stream.encode(converted):
                    destination.mux(packet)
                if frame_count % 250 == 0:
                    progress(f"Encoded {frame_count} frames")

            for packet in output_stream.encode():
                destination.mux(packet)

    if frame_count == 0:
        raise VideoConvertError(f"Source video contains no decodable frames: {source_path}")

    output_duration = frame_count / float(rate)
    # The labeling contract assumes CFR.  Silently changing either FPS or
    # duration would shift overlays, so reject a genuinely variable timeline.
    if source_duration > 0:
        duration_tolerance = max(1.0 / float(rate), 0.050) + 1e-6
        if abs(output_duration - source_duration) > duration_tolerance:
            raise VideoConvertError(
                "Source timeline is not consistent with its nominal FPS: "
                f"metadata duration={source_duration:.6f}s, "
                f"decoded frames/FPS={output_duration:.6f}s. "
                "Conversion was not installed because it could misalign annotations."
            )

    return {
        "codec": codec,
        "pixel_format": "yuv420p",
        "width": width,
        "height": height,
        "fps": float(rate),
        "fps_fraction": f"{rate.numerator}/{rate.denominator}",
        "frame_count": frame_count,
        "duration_seconds": output_duration,
    }


def _validated_paths(input_path: Path, output_path: Path) -> tuple[Path, Path]:
    source = input_path.expanduser().resolve()
    target = output_path.expanduser().resolve()
    if not source.is_file():
        raise VideoConvertError(f"Input video does not exist: {source}")
    if target.suffix.lower() != ".mp4":
        raise VideoConvertError("Output video must use the .mp4 extension")
    if source == target:
        raise VideoConvertError("Output must not overwrite the source video")
    if source.stem != target.stem:
        raise VideoConvertError(
            "Output must keep the source stem so Control/project matching remains valid: "
            f"expected {source.stem}.mp4"
        )
    if target.exists() and not target.is_file():
        raise VideoConvertError(f"Output path exists and is not a regular file: {target}")
    return source, target


def _stream_rate(stream: av.video.stream.VideoStream, source_path: Path) -> Fraction:
    raw_rate = stream.average_rate or stream.guessed_rate
    if raw_rate is None:
        raise VideoConvertError(f"Source video has no usable FPS: {source_path}")
    rate = Fraction(raw_rate.numerator, raw_rate.denominator)
    if not math.isfinite(float(rate)) or float(rate) <= 0:
        raise VideoConvertError(f"Source video has invalid FPS: {float(rate)}")
    return rate


def _duration_seconds(container: av.container.InputContainer, stream: Any) -> float:
    if stream.duration is not None and stream.time_base is not None:
        duration = float(stream.duration * stream.time_base)
    elif container.duration is not None:
        duration = float(container.duration / av.time_base)
    else:
        return 0.0
    return duration if math.isfinite(duration) and duration > 0 else 0.0


def _validate_encoded_video(output_path: Path, *, expected: dict[str, Any]) -> None:
    try:
        with av.open(str(output_path), mode="r") as container:
            video_streams = [stream for stream in container.streams if stream.type == "video"]
            audio_streams = [stream for stream in container.streams if stream.type == "audio"]
            if len(video_streams) != 1:
                raise VideoConvertError("Converted MP4 must contain exactly one video stream")
            if audio_streams:
                raise VideoConvertError("Converted MP4 unexpectedly contains an audio stream")
            stream = video_streams[0]
            codec_name = stream.codec_context.name
            if codec_name != "h264":
                raise VideoConvertError(
                    f"Converted MP4 codec is {codec_name!r}, expected browser-compatible 'h264'"
                )
            if stream.codec_context.format is None or stream.codec_context.format.name != "yuv420p":
                actual = stream.codec_context.format.name if stream.codec_context.format else "unknown"
                raise VideoConvertError(
                    f"Converted MP4 pixel format is {actual!r}, expected 'yuv420p'"
                )
            if (stream.codec_context.width, stream.codec_context.height) != (
                expected["width"],
                expected["height"],
            ):
                raise VideoConvertError("Converted MP4 dimensions do not match the source")
            rate = stream.average_rate or stream.guessed_rate
            if rate is None or abs(float(rate) - expected["fps"]) > 1e-6:
                raise VideoConvertError("Converted MP4 FPS does not match the source")
            duration = _duration_seconds(container, stream)
            tolerance = max(1.0 / expected["fps"], 0.050) + 1e-6
            if duration <= 0 or abs(duration - expected["duration_seconds"]) > tolerance:
                raise VideoConvertError(
                    "Converted MP4 duration does not match the decoded source timeline"
                )
    except VideoConvertError:
        raise
    except (av.error.FFmpegError, OSError) as exc:
        raise VideoConvertError(f"Converted MP4 failed validation: {exc}") from exc


def _temporary_mp4(target: Path) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=f".{target.stem}.",
        suffix=".tmp.mp4",
        dir=target.parent,
    )
    os.close(descriptor)
    return Path(name)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
