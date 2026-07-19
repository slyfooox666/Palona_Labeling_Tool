from __future__ import annotations

import json
from pathlib import Path

import pytest

from palona_depth.control import ControlDataError, sampled_control_frames


def test_streams_and_samples_control_frames(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    path.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frame_index": index,
                        "timestamp_seconds": index / 10,
                        "tracks": [
                            {
                                "track_id": "p0:1",
                                "label": "person",
                                "contours_xy": [[[0, 0], [10, 0], [10, 10], [0, 10]]],
                            }
                        ],
                    }
                    for index in range(11)
                ]
            }
        ),
        encoding="utf-8",
    )
    frames = sampled_control_frames(path, sample_fps=2, video_fps=10)
    assert [frame.frame_index for frame in frames] == [0, 5, 10]
    assert frames[0].tracks[0].track_id == "p0:1"


def test_rejects_git_lfs_pointer(tmp_path: Path) -> None:
    path = tmp_path / "control.json"
    path.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        "size 123\n",
        encoding="utf-8",
    )
    with pytest.raises(ControlDataError, match="Git LFS pointer"):
        sampled_control_frames(path, sample_fps=2, video_fps=10)


def test_accepts_shared_runtime_sam3_instances(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frame_index": 0,
                        "source_frame_index": 40,
                        "timestamp_seconds": 0,
                        "instances": [
                            {
                                "instance_id": "7",
                                "label": "person",
                                "score": 0.9,
                                "contours": [[[1, 1], [8, 1], [8, 8], [1, 8]]],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    frames = sampled_control_frames(path, sample_fps=1, video_fps=10)
    assert frames[0].frame_index == 40
    assert frames[0].tracks[0].track_id == "7"
    assert frames[0].tracks[0].confidence == 0.9


def test_rejects_duplicate_track_ids(tmp_path: Path) -> None:
    path = tmp_path / "duplicates.json"
    track = {
        "track_id": "same",
        "label": "person",
        "contours_xy": [[[0, 0], [10, 0], [10, 10], [0, 10]]],
    }
    path.write_text(
        json.dumps({"frames": [{"frame_index": 0, "timestamp_seconds": 0, "tracks": [track, track]}]}),
        encoding="utf-8",
    )
    with pytest.raises(ControlDataError, match="duplicate track IDs"):
        sampled_control_frames(path, sample_fps=1, video_fps=10)


def test_rejects_out_of_order_source_frame_indexes(tmp_path: Path) -> None:
    path = tmp_path / "out-of-order.json"
    path.write_text(
        json.dumps(
            {
                "frames": [
                    {"frame_index": 0, "source_frame_index": 20, "timestamp_seconds": 0, "tracks": []},
                    {"frame_index": 1, "source_frame_index": 10, "timestamp_seconds": 1, "tracks": []},
                ]
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ControlDataError, match="strictly ordered by frame_index"):
        sampled_control_frames(path, sample_fps=1, video_fps=10)
