from __future__ import annotations

import sys
import tempfile
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from vision_pipeline.utils.video import (
    extract_frames_at_fps,
    extract_sample_frames,
    probe_video,
    video_metadata_from_ffprobe,
)


def test_line_velocity_sample_videos_exist_and_probe() -> None:
    root = Path(__file__).resolve().parents[2]
    video_dir = root / "agent" / "vision" / "pokeworks" / "line-velocity"
    videos = sorted(video_dir.glob("*.mp4"))

    assert len(videos) >= 3
    for video in videos[:3]:
        metadata = probe_video(video)
        assert metadata.width > 0
        assert metadata.height > 0
        assert metadata.duration_seconds > 0


def test_probe_video_metadata_uses_format_duration_fallback() -> None:
    metadata = video_metadata_from_ffprobe(
        Path("clip.mkv"),
        {
            "streams": [
                {
                    "width": 2560,
                    "height": 1440,
                    "avg_frame_rate": "20/1",
                }
            ],
            "format": {"duration": "60.000000"},
        },
    )

    assert metadata.duration_seconds == 60.0
    assert metadata.fps == 20.0


def test_line_velocity_sample_video_frame_extraction() -> None:
    root = Path(__file__).resolve().parents[2]
    video = root / "agent" / "vision" / "pokeworks" / "line-velocity" / "2026-05-27_18-07-07.mp4"

    with tempfile.TemporaryDirectory() as temp_dir:
        frames = extract_sample_frames(video, temp_dir, sample_count=2)

        assert len(frames) == 2
        assert frames[0].path.exists()
        assert frames[1].path.exists()
        assert frames[0].timestamp_seconds < frames[1].timestamp_seconds


def test_frame_extraction_supports_start_offset_and_workers() -> None:
    root = Path(__file__).resolve().parents[2]
    video = root / "agent" / "vision" / "pokeworks" / "line-velocity" / "2026-05-27_18-07-07.mp4"

    with tempfile.TemporaryDirectory() as temp_dir:
        frames = extract_frames_at_fps(
            video,
            temp_dir,
            sample_fps=1.0,
            max_frames=2,
            start_from_seconds=3.0,
            worker_threads=2,
        )

        assert [frame.timestamp_seconds for frame in frames] == [3.0, 4.0]
        assert all(frame.path.exists() for frame in frames)
