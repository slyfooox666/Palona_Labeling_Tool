from __future__ import annotations

import hashlib
from pathlib import Path

import av
import numpy as np
import pytest

from palona_depth.video_convert import VideoConvertError, VideoConvertOptions, convert_video


def write_mkv(path: Path, *, width: int = 64, height: int = 48, frames: int = 7) -> None:
    with av.open(str(path), "w", format="matroska") as container:
        stream = container.add_stream("mpeg4", rate=5)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        for index in range(frames):
            image = np.full((height, width, 3), 20 + index * 20, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_video(path: Path) -> dict[str, object]:
    with av.open(str(path)) as container:
        stream = next(item for item in container.streams if item.type == "video")
        decoded = list(container.decode(stream))
        duration = float(stream.duration * stream.time_base)
        return {
            "codec": stream.codec_context.name,
            "format": stream.codec_context.format.name,
            "width": stream.codec_context.width,
            "height": stream.codec_context.height,
            "fps": float(stream.average_rate),
            "duration": duration,
            "frames": len(decoded),
            "audio_streams": len([item for item in container.streams if item.type == "audio"]),
        }


def test_converts_mkv_to_same_stem_browser_mp4_without_mutating_source(tmp_path: Path) -> None:
    source = tmp_path / "scene.mkv"
    target = tmp_path / "scene.mp4"
    write_mkv(source)
    source_digest = sha256(source)
    progress: list[str] = []

    result = convert_video(
        VideoConvertOptions(input_path=source, output_path=target),
        progress=progress.append,
    )

    assert sha256(source) == source_digest
    assert result["audio_preserved"] is False
    assert result["frame_count"] == 7
    assert result["fps"] == 5.0
    assert result["duration_seconds"] == pytest.approx(1.4)
    assert progress[0].startswith("Converting scene.mkv")
    assert progress[-1] == "Video conversion complete"

    encoded = inspect_video(target)
    assert encoded == {
        "codec": "h264",
        "format": "yuv420p",
        "width": 64,
        "height": 48,
        "fps": 5.0,
        "duration": pytest.approx(1.4),
        "frames": 7,
        "audio_streams": 0,
    }


def test_failed_conversion_keeps_existing_output_and_cleans_temporary_file(tmp_path: Path) -> None:
    source = tmp_path / "scene.mkv"
    target = tmp_path / "scene.mp4"
    write_mkv(source)
    target.write_bytes(b"previous-good-output")

    with pytest.raises(VideoConvertError, match="unavailable"):
        convert_video(
            VideoConvertOptions(
                input_path=source,
                output_path=target,
                codec="encoder-that-does-not-exist",
            )
        )

    assert target.read_bytes() == b"previous-good-output"
    assert list(tmp_path.glob(".scene.*.tmp.mp4")) == []


@pytest.mark.parametrize(
    ("output_name", "message"),
    [
        ("other.mp4", "keep the source stem"),
        ("scene.mov", r"\.mp4 extension"),
    ],
)
def test_rejects_unsafe_or_mismatched_output(
    tmp_path: Path,
    output_name: str,
    message: str,
) -> None:
    source = tmp_path / "scene.mkv"
    write_mkv(source)

    with pytest.raises(VideoConvertError, match=message):
        convert_video(VideoConvertOptions(input_path=source, output_path=tmp_path / output_name))


def test_refuses_to_resize_odd_dimensions_for_yuv420p(tmp_path: Path) -> None:
    source = tmp_path / "scene.mkv"
    target = tmp_path / "scene.mp4"
    # yuv444p permits odd dimensions, allowing the converter guard to be tested.
    with av.open(str(source), "w", format="matroska") as container:
        stream = container.add_stream("ffv1", rate=5)
        stream.width = 63
        stream.height = 47
        stream.pix_fmt = "yuv444p"
        frame = av.VideoFrame.from_ndarray(np.zeros((47, 63, 3), dtype=np.uint8), format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)

    with pytest.raises(VideoConvertError, match="requires even source dimensions"):
        convert_video(VideoConvertOptions(input_path=source, output_path=target))
    assert not target.exists()
