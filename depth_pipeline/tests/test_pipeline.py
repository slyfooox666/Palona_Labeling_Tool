from __future__ import annotations

import json
import math
from pathlib import Path

import av
from jsonschema import Draft202012Validator
import numpy as np

from palona_depth.models import DepthArtifact
from palona_depth.pipeline import BuildOptions, assert_artifacts_outside_git, build_depth_features
from palona_depth.safety import require_private_json_suffix


class FakeDepthClient:
    def __init__(self) -> None:
        self.stopped = False

    def ensure_ready(self):  # noqa: ANN201
        return {
            "device": "cpu",
            "dtype": "float32",
            "model": {
                "model_name": "depth-anything/DA3-BASE",
                "model_revision": "f4a6c9b3c95e41c82048423d3493a81ec3fa810e",
            },
        }

    def infer_image(self, input_path: Path, output_dir: Path) -> DepthArtifact:
        output_dir.mkdir(parents=True, exist_ok=True)
        depth = np.tile(np.linspace(1, 9, 64, dtype=np.float32), (48, 1))
        confidence = np.ones_like(depth)
        depth_path = output_dir / "depth.npy"
        confidence_path = output_dir / "confidence.npy"
        np.save(depth_path, depth)
        np.save(confidence_path, confidence)
        return DepthArtifact(
            depth_path=depth_path,
            confidence_path=confidence_path,
            shape=depth.shape,
            model={
                "model_name": "depth-anything/DA3-BASE",
                "model_revision": "f4a6c9b3c95e41c82048423d3493a81ec3fa810e",
                "device": "cpu",
                "dtype": "float32",
            },
            processing={"total_seconds": 0.01},
        )

    def stop(self) -> None:
        self.stopped = True


def write_video(path: Path) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        for index in range(8):
            image = np.zeros((48, 64, 3), dtype=np.uint8)
            image[:, :, 0] = index * 20
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def write_control(path: Path) -> None:
    frames = []
    for index in range(8):
        person_x = 4 + index * 3
        frames.append(
            {
                "frame_index": index,
                "timestamp_seconds": index / 4,
                "tracks": [
                    {
                        "track_id": "p0:1",
                        "label": "person",
                        "confidence": 0.95,
                        "contours_xy": [
                            [[person_x, 10], [person_x + 10, 10], [person_x + 10, 38], [person_x, 38]]
                        ],
                    },
                    {
                        "track_id": "p1:0",
                        "label": "table",
                        "confidence": 0.98,
                        "contours_xy": [[[38, 8], [60, 8], [60, 42], [38, 42]]],
                    },
                ],
            }
        )
    path.write_text(json.dumps({"video": "synthetic.mp4", "frames": frames}), encoding="utf-8")


def test_end_to_end_builds_versioned_sidecar(tmp_path: Path) -> None:
    video = tmp_path / "synthetic.mp4"
    contour = tmp_path / "synthetic.json"
    output = tmp_path / "synthetic.depth-features.json"
    write_video(video)
    write_control(contour)
    client = FakeDepthClient()
    payload = build_depth_features(
        BuildOptions(
            video_path=video,
            contour_path=contour,
            output_path=output,
            sample_fps=4,
            alignment_tolerance_seconds=0.13,
            stop_runtime=True,
        ),
        client=client,
    )
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored == payload
    schema_path = Path(__file__).parents[2] / "docs" / "depth-features.schema.json"
    Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).validate(payload)
    assert payload["schema_version"] == "palona.depth-features/v1"
    assert payload["depth_metadata"]["metric"] is False
    assert payload["depth_metadata"]["depth_semantics"] == "depth_rank: 0=near, 1=far"
    assert payload["source"]["video_width"] == 64
    assert payload["source"]["video_file_size_bytes"] == video.stat().st_size
    assert payload["source"]["contour_file_size_bytes"] == contour.stat().st_size
    assert payload["depth_metadata"]["normalization"]["method"] == "clip_robust_quantile"
    assert payload["depth_metadata"]["max_cue_age_seconds"] > 0
    assert len(payload["frames"]) == 8
    assert payload["frames"][0]["pairs"][0]["source_id"] == "p0:1"
    assert client.stopped


def test_refuses_to_overwrite_control_json(tmp_path: Path) -> None:
    video = tmp_path / "synthetic.mp4"
    contour = tmp_path / "synthetic.json"
    write_video(video)
    write_control(contour)

    try:
        build_depth_features(
            BuildOptions(video_path=video, contour_path=contour, output_path=contour),
            client=FakeDepthClient(),
        )
    except ValueError as error:
        assert "must not overwrite" in str(error)
    else:
        raise AssertionError("Expected source overwrite protection")


def test_rejects_non_positive_or_non_finite_alignment_tolerance(tmp_path: Path) -> None:
    video = tmp_path / "synthetic.mp4"
    contour = tmp_path / "synthetic.json"
    output = tmp_path / "features.json"
    write_video(video)
    write_control(contour)

    for tolerance in (0.0, -0.1, math.nan, math.inf):
        try:
            build_depth_features(
                BuildOptions(
                    video_path=video,
                    contour_path=contour,
                    output_path=output,
                    alignment_tolerance_seconds=tolerance,
                ),
                client=FakeDepthClient(),
            )
        except ValueError as error:
            assert "finite positive" in str(error)
        else:
            raise AssertionError(f"Expected invalid tolerance rejection for {tolerance}")


def test_retained_artifacts_must_stay_outside_git(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / ".git").mkdir()

    try:
        assert_artifacts_outside_git(repository / "private-depth")
    except ValueError as error:
        assert "outside every Git worktree" in str(error)
    else:
        raise AssertionError("Expected Git worktree artifact protection")


def test_depth_json_inside_git_requires_ignored_safe_suffix(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / ".git").mkdir()
    video = repository / "synthetic.mp4"
    contour = repository / "synthetic.json"
    write_video(video)
    write_control(contour)

    try:
        build_depth_features(
            BuildOptions(
                video_path=video,
                contour_path=contour,
                output_path=repository / "private-depth.json",
            ),
            client=FakeDepthClient(),
        )
    except ValueError as error:
        assert ".depth-features.json" in str(error)
        assert ".gitignore" in str(error)
    else:
        raise AssertionError("Expected unsafe generated JSON suffix rejection")

    require_private_json_suffix(
        repository / "private.depth-features.json",
        suffix=".depth-features.json",
        artifact_name="Depth sidecar output",
    )
