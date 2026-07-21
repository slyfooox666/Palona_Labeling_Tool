from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TOOLS_ROOT = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS_ROOT))

from draw_video_contours import main, validate_paths  # noqa: E402


def test_validate_paths_accepts_mkv_and_defaults_to_mp4(tmp_path: Path) -> None:
    video_path = tmp_path / "clip.mkv"
    contour_json_path = tmp_path / "clip.json"
    video_path.write_bytes(b"")
    contour_json_path.write_text('{"frames": []}', encoding="utf-8")
    args = argparse.Namespace(
        video=str(video_path),
        contour_json=str(contour_json_path),
        output=None,
    )

    video, contour_json, output = validate_paths(args)

    assert video == video_path
    assert contour_json == contour_json_path
    assert output == tmp_path / "clip_contours.mp4"


def test_main_passes_compression_options_to_drawer(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    video_path = tmp_path / "clip.mkv"
    contour_json_path = tmp_path / "clip.json"
    output_path = tmp_path / "annotated.mp4"
    video_path.write_bytes(b"")
    contour_json_path.write_text('{"frames": []}', encoding="utf-8")
    calls = {}

    def fake_draw_serialized_contour_video(
        source_video_path,
        contour_json_path,
        output_path,
        **kwargs,
    ):
        calls["source_video_path"] = source_video_path
        calls["contour_json_path"] = contour_json_path
        calls["output_path"] = output_path
        calls["kwargs"] = kwargs
        return {
            "frames_read": 1,
            "annotated_frames": 1,
            "drawn_annotations": 1,
        }

    monkeypatch.setattr(
        "draw_video_contours.draw_serialized_contour_video",
        fake_draw_serialized_contour_video,
    )

    assert (
        main(
            [
                str(video_path),
                str(contour_json_path),
                "--output",
                str(output_path),
                "--crf",
                "28",
                "--preset",
                "fast",
            ]
        )
        == 0
    )

    assert calls["source_video_path"] == video_path
    assert calls["contour_json_path"] == contour_json_path
    assert calls["output_path"] == output_path
    assert calls["kwargs"]["crf"] == 28
    assert calls["kwargs"]["preset"] == "fast"
    payload = json.loads(capsys.readouterr().out)
    assert payload["output"] == str(output_path)
