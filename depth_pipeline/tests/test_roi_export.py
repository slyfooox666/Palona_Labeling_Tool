from __future__ import annotations

import hashlib
import json
from pathlib import Path

import av
import numpy as np
import pytest

from palona_depth.roi_export import (
    RoiExportError,
    RoiExportOptions,
    export_roi,
    load_normalized_roi,
    validate_normalized_polygon,
)


ROI = [[0.0, 0.0], [0.48, 0.0], [0.48, 1.0], [0.0, 1.0]]


def write_video(path: Path) -> None:
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=4)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        for index in range(4):
            image = np.full((48, 64, 3), (180 + index * 10), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def control_track(track_id: str, contour: list[list[float]]) -> dict[str, object]:
    return {
        "track_id": track_id,
        "label": "person",
        "confidence": 0.9,
        "contours_xy": [contour],
    }


def write_control(path: Path) -> None:
    inside = [[3, 5], [16, 5], [16, 35], [3, 35]]
    outside = [[45, 5], [60, 5], [60, 35], [45, 35]]
    crossing_centroid_outside = [[24, 5], [50, 5], [50, 35], [24, 35]]
    frames = []
    for index in range(4):
        multi = control_track("multi", outside)
        multi["contours_xy"] = [outside, inside]
        frames.append(
            {
                "frame_index": index,
                "timestamp_seconds": index / 4,
                "tracks": [
                    control_track("inside", inside),
                    control_track("outside", outside),
                    control_track("crossing", crossing_centroid_outside),
                    multi,
                ],
            }
        )
    path.write_text(json.dumps({"video": "source.mp4", "frames": frames}), encoding="utf-8")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decoded_images(path: Path) -> list[np.ndarray]:
    with av.open(str(path)) as container:
        stream = next(item for item in container.streams if item.type == "video")
        return [frame.to_ndarray(format="rgb24") for frame in container.decode(stream)]


def test_roi_export_blacks_outside_filters_tracks_and_preserves_sources(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    control = tmp_path / "control.json"
    project = tmp_path / "project.json"
    masked = tmp_path / "exports" / "masked.mp4"
    filtered = tmp_path / "exports" / "filtered.json"
    write_video(video)
    write_control(control)
    project.write_text(json.dumps({"version": 1, "roi": {"polygon": ROI}}), encoding="utf-8")
    source_digests = {path: digest(path) for path in (video, control, project)}

    result = export_roi(
        RoiExportOptions(
            video_path=video,
            contour_path=control,
            roi_path=project,
            masked_video_path=masked,
            filtered_contour_path=filtered,
        )
    )

    assert result["video_frame_count"] == 4
    assert result["control_frame_count"] == 4
    assert result["kept_track_appearances"] == 8
    assert result["removed_track_appearances"] == 8
    assert all(digest(path) == expected for path, expected in source_digests.items())

    images = decoded_images(masked)
    assert len(images) == 4
    assert float(images[0][:, 5].mean()) > 140
    assert int(images[0][:, 55].max()) < 20

    payload = json.loads(filtered.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "palona.filtered-contours/v1"
    assert payload["video"] == str(masked.resolve())
    assert [track["track_id"] for track in payload["frames"][0]["tracks"]] == ["inside", "multi"]
    assert len(payload["frames"][0]["tracks"][1]["contours_xy"]) == 2


def test_runtime_instances_are_normalized_to_tracks(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    control = tmp_path / "runtime.json"
    roi = tmp_path / "roi.json"
    write_video(video)
    roi.write_text(json.dumps(ROI), encoding="utf-8")
    control.write_text(
        json.dumps(
            {
                "frames": [
                    {
                        "frame_index": 0,
                        "timestamp_seconds": 0.0,
                        "instances": [
                            {
                                "instance_id": "7",
                                "prompt_label": "person",
                                "score": 0.88,
                                "contours": [[[2, 2], [12, 2], [12, 18], [2, 18]]],
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    masked = tmp_path / "masked.mp4"
    filtered = tmp_path / "filtered.json"

    export_roi(
        RoiExportOptions(
            video_path=video,
            contour_path=control,
            roi_path=roi,
            masked_video_path=masked,
            filtered_contour_path=filtered,
        )
    )
    track = json.loads(filtered.read_text(encoding="utf-8"))["frames"][0]["tracks"][0]
    assert track["track_id"] == "7"
    assert track["label"] == "person"
    assert track["confidence"] == 0.88


def test_project_can_reference_or_sit_beside_roi_json(tmp_path: Path) -> None:
    roi = tmp_path / "roi.json"
    roi.write_text(json.dumps({"normalized_polygon": ROI}), encoding="utf-8")
    referenced = tmp_path / "referenced.json"
    referenced.write_text(json.dumps({"roi_path": "roi.json"}), encoding="utf-8")
    sibling_project = tmp_path / "project.json"
    sibling_project.write_text(json.dumps({"version": "1"}), encoding="utf-8")

    assert load_normalized_roi(referenced) == load_normalized_roi(roi)
    assert load_normalized_roi(sibling_project) == load_normalized_roi(roi)


@pytest.mark.parametrize(
    ("polygon", "message"),
    [
        ([[0, 0], [1, 0]], "at least three"),
        ([[0, 0], [1.1, 0], [0, 1]], "normalized range"),
        ([[0, 0], [1, 1], [0, 1], [1, 0]], "self-intersect"),
        ([[0, 0], [0.5, 0.5], [1, 1]], "area"),
    ],
)
def test_rejects_invalid_normalized_polygons(polygon: list[list[float]], message: str) -> None:
    with pytest.raises(RoiExportError, match=message):
        validate_normalized_polygon(polygon)


def test_refuses_to_overwrite_any_source(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    control = tmp_path / "control.json"
    roi = tmp_path / "roi.json"
    write_video(video)
    write_control(control)
    roi.write_text(json.dumps(ROI), encoding="utf-8")

    with pytest.raises(RoiExportError, match="must not overwrite"):
        export_roi(
            RoiExportOptions(
                video_path=video,
                contour_path=control,
                roi_path=roi,
                masked_video_path=video,
                filtered_contour_path=tmp_path / "filtered.json",
            )
        )


def test_filtered_contour_inside_git_requires_ignored_safe_suffix(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / ".git").mkdir()
    video = repository / "source.mp4"
    control = repository / "control.json"
    roi = repository / "roi.json"
    masked = repository / "masked.mp4"
    write_video(video)
    write_control(control)
    roi.write_text(json.dumps(ROI), encoding="utf-8")

    with pytest.raises(RoiExportError, match=r"\.filtered-contours\.json"):
        export_roi(
            RoiExportOptions(
                video_path=video,
                contour_path=control,
                roi_path=roi,
                masked_video_path=masked,
                filtered_contour_path=repository / "filtered.json",
            )
        )

    safe_filtered = repository / "scene.filtered-contours.json"
    export_roi(
        RoiExportOptions(
            video_path=video,
            contour_path=control,
            roi_path=roi,
            masked_video_path=masked,
            filtered_contour_path=safe_filtered,
        )
    )
    assert masked.is_file()
    assert safe_filtered.is_file()
