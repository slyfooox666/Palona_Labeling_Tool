from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

TOOLS_ROOT = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS_ROOT))

from side_by_side_video import (  # noqa: E402
    CropSpec,
    ToolError,
    build_ffmpeg_command,
    build_filter_complex,
    default_output_path,
    validate_inputs,
)


def test_default_output_path_uses_both_video_stems() -> None:
    output_path = default_output_path(Path("left.mp4"), Path("right.mp4"))

    assert output_path == Path("left_right_side_by_side.mp4")


def test_build_filter_complex_adds_titles_and_hstack() -> None:
    filter_complex = build_filter_complex(
        left_width=160,
        right_width=120,
        height=120,
        fps="10",
        left_crop=CropSpec(10, 20, 90, 100),
        title_margin_y=18,
        shortest=True,
    )

    assert "[0:v]crop=w=80:h=80:x=10:y=20,scale=w=160:h=120" in filter_complex
    assert "[2:v]setsar=1,format=rgba[leftt]" in filter_complex
    assert "[3:v]setsar=1,format=rgba[rightt]" in filter_complex
    assert "scale=w=160:h=120" in filter_complex
    assert "scale=w=120:h=120" in filter_complex
    assert (
        "overlay=x=(main_w-overlay_w)/2:y=18:shortest=1:format=auto"
        in filter_complex
    )
    assert "hstack=inputs=2:shortest=1" in filter_complex
    assert filter_complex.endswith("fps=10,format=yuv420p[v]")


def test_build_ffmpeg_command_maps_requested_audio() -> None:
    command = build_ffmpeg_command(
        Path("left.mp4"),
        Path("right.mp4"),
        Path("left_title.png"),
        Path("right_title.png"),
        Path("out.mp4"),
        filter_complex="[0:v][1:v]hstack[v]",
        title_framerate="10",
        audio="right",
        crf=18,
        preset="medium",
        overwrite=True,
    )

    assert "-map" in command
    assert "1:a:0?" in command
    assert "-shortest" in command
    assert "-an" not in command


def test_validate_inputs_rejects_output_overwriting_input(tmp_path) -> None:
    left = tmp_path / "left.mp4"
    right = tmp_path / "right.mp4"
    left.write_text("", encoding="utf-8")
    right.write_text("", encoding="utf-8")
    args = argparse.Namespace(
        left_video=str(left),
        right_video=str(right),
        output=str(left),
        overwrite=False,
        title_height=64,
        font_size=32,
        title_margin_y=24,
    )

    with pytest.raises(ToolError):
        validate_inputs(args)
