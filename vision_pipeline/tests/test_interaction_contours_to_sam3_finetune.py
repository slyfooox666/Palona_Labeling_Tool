from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TOOLS_ROOT = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS_ROOT))

from interaction_contours_to_sam3_finetune import (  # noqa: E402
    ExportOptions,
    build_object_summaries,
    export_dataset,
    parse_frame_annotations,
    render_indexed_mask,
)


def test_render_indexed_mask_fills_all_contours_for_track() -> None:
    frames = parse_frame_annotations(
        {
            "frames": [
                {
                    "frame_index": 0,
                    "timestamp_seconds": 0.0,
                    "tracks": [
                        {
                            "track_id": "i1",
                            "label": "event",
                            "state": "active",
                            "contours_xy": [
                                [[1, 1], [4, 1], [4, 4], [1, 4]],
                                [[6, 6], [8, 6], [8, 8], [6, 8]],
                            ],
                        }
                    ],
                }
            ]
        }
    )
    objects = build_object_summaries(frames)

    mask = render_indexed_mask(
        frames[0],
        {track_id: summary.object_id for track_id, summary in objects.items()},
        width=10,
        height=10,
    )

    assert mask[2, 2] == 1
    assert mask[7, 7] == 1
    assert mask[0, 0] == 0


def test_export_dataset_writes_vos_layout(tmp_path: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    video_path = tmp_path / "clip.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (16, 16),
    )
    if not writer.isOpened():
        pytest.skip("OpenCV cannot create an mp4 test video in this environment")
    for value in (30, 60, 90):
        writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
    writer.release()

    interaction_path = tmp_path / "clip_interaction.json"
    interaction_path.write_text(
        json.dumps(
            {
                "video": str(video_path),
                "frames": [
                    {"frame_index": 0, "timestamp_seconds": 0.0, "tracks": []},
                    {
                        "frame_index": 1,
                        "timestamp_seconds": 0.2,
                        "tracks": [
                            {
                                "track_id": "i1",
                                "label": "pay",
                                "confidence": 1.0,
                                "state": "active",
                                "bbox_xyxy": [2, 2, 12, 12],
                                "contours_xy": [
                                    [[2, 2], [8, 2], [8, 8], [2, 8]],
                                    [[10, 10], [12, 10], [12, 12], [10, 12]],
                                ],
                                "contours_format": "absolute_xy",
                            }
                        ],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "sam3_finetune"

    summary = export_dataset(
        [interaction_path],
        ExportOptions(
            output_dir=output_dir,
            include_empty_frames=False,
            image_ext=".jpg",
            image_quality=80,
            preview_mask_dir=output_dir / "AnnotationPreviews",
            overwrite=False,
            dry_run=False,
            video_root=None,
        ),
    )

    assert summary["sequences"] == 1
    assert summary["frames"] == 1
    assert (output_dir / "JPEGImages" / "clip" / "000001.jpg").exists()
    mask_path = output_dir / "Annotations" / "clip" / "000001.png"
    assert mask_path.exists()
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    assert mask[4, 4] == 1
    assert mask[11, 11] == 1
    assert mask[0, 0] == 0
    preview = cv2.imread(
        str(output_dir / "AnnotationPreviews" / "clip" / "000001.png"),
        cv2.IMREAD_COLOR,
    )
    assert preview[4, 4].max() > 200
    assert preview[0, 0].max() == 0

    meta = json.loads((output_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["format"] == "sam3_vos_indexed_masks"
    assert meta["sequences"][0]["objects"][0]["track_id"] == "i1"
    assert (output_dir / "train.txt").read_text(encoding="utf-8") == "clip\n"
    manifest_lines = (output_dir / "manifest.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(manifest_lines) == 1
    assert json.loads(manifest_lines[0])["objects"][0]["label"] == "pay"
