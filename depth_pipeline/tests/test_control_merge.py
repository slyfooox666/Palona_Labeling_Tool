from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from palona_depth.control import sampled_control_frames
from palona_depth.control_merge import (
    ControlMergeError,
    ManifestSource,
    merge_control_manifests,
)


VIDEO = "/private/research/camera-01.mkv"


def manifest(prompt: str, *, instance_id: str, frame_indexes: tuple[int, ...] = (0, 20)) -> dict:
    return {
        "schema_version": "ai-model-runtime.sam3/v1",
        "task": "sam3.track_video",
        "input_path": VIDEO,
        "model": {
            "model_name": "facebook/sam3",
            "model_revision": "revision-1",
            "device": "cpu",
            "dtype": "float32",
        },
        "media": {
            "width": 100,
            "height": 80,
            "source_fps": 20.0,
            "sample_fps": 1.0,
            "sampled_frames": len(frame_indexes),
        },
        "frames": [
            {
                "frame_index": ordinal,
                "source_frame_index": frame_index,
                "timestamp_seconds": float(ordinal),
                "instances": [
                    {
                        "instance_id": instance_id,
                        "label": prompt,
                        "score": 0.9 - ordinal * 0.1,
                        "bbox_xyxy": [10, 10, 30, 40],
                        "contours": [[[10, 10], [30, 10], [30, 40], [10, 40]]],
                    }
                ],
            }
            for ordinal, frame_index in enumerate(frame_indexes)
        ],
    }


def write_manifest(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_merges_and_namespaces_runtime_manifests(tmp_path: Path) -> None:
    person_path = write_manifest(tmp_path / "person.json", manifest("person", instance_id="7"))
    table_path = write_manifest(tmp_path / "table.json", manifest("table", instance_id="7"))
    output = tmp_path / "clip.control.json"

    summary = merge_control_manifests(
        [ManifestSource("person", person_path), ManifestSource("table", table_path)],
        output,
    )

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert summary.frame_count == 2
    assert summary.track_count == 4
    assert payload["schema_version"] == "palona.control/v1"
    assert payload["media"] == {
        "width": 100,
        "height": 80,
        "source_fps": 20.0,
        "sample_fps": 1.0,
        "sampled_frames": 2,
    }
    assert [track["track_id"] for track in payload["frames"][0]["tracks"]] == ["p0:7", "p1:7"]
    assert payload["frames"][0]["tracks"][0]["confidence"] == 0.9
    assert payload["frames"][0]["tracks"][0]["bbox_xyxy"] == [10.0, 10.0, 30.0, 40.0]
    assert payload["frames"][0]["tracks"][0]["contours_xy"][0][0] == [10.0, 10.0]

    normalized = sampled_control_frames(output, sample_fps=1, video_fps=20)
    assert [frame.frame_index for frame in normalized] == [0, 20]
    assert [track.track_id for track in normalized[0].tracks] == ["p0:7", "p1:7"]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("input_path", "/private/research/other.mkv"), "canonical_input_path mismatch"),
        (("media.width", 101), "width mismatch"),
        (("media.height", 81), "height mismatch"),
        (("media.source_fps", 25.0), "source_fps mismatch"),
        (("model.model_revision", "revision-2"), "model_revision mismatch"),
        (("frames.1.source_frame_index", 21), "source_frame_index mismatch"),
        (("frames.1.timestamp_seconds", 1.01), "timestamp_seconds mismatch"),
    ],
)
def test_rejects_source_metadata_or_frame_mismatch(
    tmp_path: Path,
    mutation: tuple[str, object],
    message: str,
) -> None:
    person = manifest("person", instance_id="1")
    table = manifest("table", instance_id="2")
    key, value = mutation
    target: object = table
    parts = key.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]  # type: ignore[index]
    if isinstance(target, list):
        target[int(parts[-1])] = value
    else:
        target[parts[-1]] = value  # type: ignore[index]
    person_path = write_manifest(tmp_path / "person.json", person)
    table_path = write_manifest(tmp_path / "table.json", table)

    with pytest.raises(ControlMergeError, match=message):
        merge_control_manifests(
            [ManifestSource("person", person_path), ManifestSource("table", table_path)],
            tmp_path / "bad.control.json",
        )


def test_rejects_different_frame_counts_atomically(tmp_path: Path) -> None:
    person_path = write_manifest(tmp_path / "person.json", manifest("person", instance_id="1"))
    table_payload = manifest("table", instance_id="2", frame_indexes=(0,))
    table_path = write_manifest(tmp_path / "table.json", table_payload)
    output = tmp_path / "existing.control.json"
    output.write_text('{"sentinel":true}', encoding="utf-8")

    with pytest.raises(ControlMergeError, match="frame counts differ"):
        merge_control_manifests(
            [ManifestSource("person", person_path), ManifestSource("table", table_path)],
            output,
        )

    assert json.loads(output.read_text(encoding="utf-8")) == {"sentinel": True}
    assert not list(tmp_path.glob(".existing.control.json.*.tmp"))


def test_rejects_overwriting_an_input_manifest(tmp_path: Path) -> None:
    person_path = write_manifest(tmp_path / "person.control.json", manifest("person", instance_id="1"))
    with pytest.raises(ControlMergeError, match="must not overwrite"):
        merge_control_manifests([ManifestSource("person", person_path)], person_path)


def test_requires_safe_suffix_inside_git_worktree(tmp_path: Path) -> None:
    worktree = tmp_path / "repo"
    worktree.mkdir()
    (worktree / ".git").mkdir()
    source = write_manifest(tmp_path / "person.json", manifest("person", instance_id="1"))

    with pytest.raises(ValueError, match=r"\.control\.json"):
        merge_control_manifests([ManifestSource("person", source)], worktree / "unsafe.json")


def test_rejects_prompt_label_mismatch_without_replacing_output(tmp_path: Path) -> None:
    payload = copy.deepcopy(manifest("person", instance_id="1"))
    source = write_manifest(tmp_path / "person.json", payload)
    output = tmp_path / "result.control.json"
    output.write_text('{"old":true}', encoding="utf-8")

    with pytest.raises(ControlMergeError, match="does not match --source prompt"):
        merge_control_manifests([ManifestSource("table", source)], output)

    assert json.loads(output.read_text(encoding="utf-8")) == {"old": True}
