from __future__ import annotations

import json
import pickle
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path

import pytest

TOOLS_ROOT = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS_ROOT))

from sam3_finetune_dataset_to_sam3_json import (  # noqa: E402
    ConvertOptions,
    FrameConversionResult,
    MODEL_TARGET,
    SAM3_VIDEO_TRAIN_API_FROM_JSON,
    build_sam3_json,
    convert_dataset,
    iter_completed_frame_futures,
)


def write_tiny_dataset(dataset_dir: Path) -> None:
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")

    image_dir = dataset_dir / "JPEGImages" / "seq1"
    mask_dir = dataset_dir / "Annotations" / "seq1"
    image_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    frame0 = np.full((6, 8, 3), 40, dtype=np.uint8)
    frame1 = np.full((6, 8, 3), 80, dtype=np.uint8)
    mask0 = np.zeros((6, 8), dtype=np.uint8)
    mask1 = np.zeros((6, 8), dtype=np.uint8)
    mask0[1:4, 1:4] = 1
    mask1[1:4, 2:5] = 1
    mask1[3:6, 5:8] = 2

    assert cv2.imwrite(str(image_dir / "000000.jpg"), frame0)
    assert cv2.imwrite(str(image_dir / "000001.jpg"), frame1)
    assert cv2.imwrite(str(mask_dir / "000000.png"), mask0)
    assert cv2.imwrite(str(mask_dir / "000001.png"), mask1)

    meta = {
        "format": "sam3_vos_indexed_masks",
        "version": 1,
        "sequences": [
            {
                "name": "seq1",
                "source_json": "seq1_interaction.json",
                "source_video": "seq1.mp4",
                "objects": [
                    {"id": 1, "track_id": "i1", "label": "pay"},
                    {"id": 2, "track_id": "i2", "label": "pickup"},
                ],
            }
        ],
    }
    (dataset_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    manifest_records = [
        {
            "sequence": "seq1",
            "frame_index": 0,
            "image": "JPEGImages/seq1/000000.jpg",
            "mask": "Annotations/seq1/000000.png",
            "width": 8,
            "height": 6,
            "objects": [{"id": 1, "track_id": "i1", "label": "pay"}],
        },
        {
            "sequence": "seq1",
            "frame_index": 1,
            "image": "JPEGImages/seq1/000001.jpg",
            "mask": "Annotations/seq1/000001.png",
            "width": 8,
            "height": 6,
            "objects": [
                {"id": 1, "track_id": "i1", "label": "pay"},
                {"id": 2, "track_id": "i2", "label": "pickup"},
            ],
        },
    ]
    with (dataset_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in manifest_records:
            handle.write(json.dumps(record) + "\n")
    (dataset_dir / "train.txt").write_text("seq1\n", encoding="utf-8")


def test_build_sam3_json_from_indexed_masks(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "sam3_finetune"
    write_tiny_dataset(dataset_dir)

    data = build_sam3_json(dataset_dir, show_progress=False)

    assert data["format"] == "pal_sam3_video_train_json"
    assert data["sample_unit"] == "frame_category"
    assert [category["name"] for category in data["categories"]] == ["pay", "pickup"]
    assert len(data["videos"]) == 1
    assert data["videos"][0]["file_names"] == [
        "JPEGImages/seq1/000000.jpg",
        "JPEGImages/seq1/000001.jpg",
    ]
    assert len(data["annotations"]) == 3
    assert len(data["video_np_pairs"]) == 4
    assert len(data["frame_category_pairs"]) == 4
    assert len(data["video_object_pairs"]) == 2
    assert [pair["is_positive"] for pair in data["frame_category_pairs"]] == [
        True,
        False,
        True,
        True,
    ]
    assert [pair["annotation_count"] for pair in data["frame_category_pairs"]] == [
        1,
        0,
        1,
        1,
    ]
    assert data["annotations"][0]["bbox"] == [1.0, 1.0, 3.0, 3.0]
    assert data["annotations"][0]["area"] == 9
    assert isinstance(data["annotations"][0]["segmentation"]["counts"], list)


def test_build_sam3_json_parallel_matches_single_process(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "sam3_finetune"
    write_tiny_dataset(dataset_dir)

    serial = build_sam3_json(dataset_dir, show_progress=False, convert_workers=1)
    parallel = build_sam3_json(dataset_dir, show_progress=False, convert_workers=2)

    assert parallel == serial


def test_iter_completed_frame_futures_yields_finished_work_first(monkeypatch) -> None:
    def fake_convert(task):
        if task == "slow":
            time.sleep(0.05)
        return FrameConversionResult(
            sequence_name="seq1",
            local_image_id=0 if task == "slow" else 1,
            frame_index=0 if task == "slow" else 1,
            image_rel=f"{task}.jpg",
            width=2,
            height=2,
            annotations=[],
        )

    monkeypatch.setattr(
        "sam3_finetune_dataset_to_sam3_json.convert_frame_record",
        fake_convert,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(iter_completed_frame_futures(executor, ["slow", "fast"], 2))

    assert [result.image_rel for result in results] == ["fast.jpg", "slow.jpg"]


def test_build_sam3_json_respects_max_frames(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "sam3_finetune"
    write_tiny_dataset(dataset_dir)

    data = build_sam3_json(dataset_dir, max_frames=1, show_progress=False)

    assert len(data["videos"]) == 1
    assert data["videos"][0]["file_names"] == ["JPEGImages/seq1/000000.jpg"]
    assert data["videos"][0]["frame_indices"] == [0]
    assert [category["name"] for category in data["categories"]] == ["pay"]
    assert len(data["annotations"]) == 1
    assert len(data["video_np_pairs"]) == 1
    assert data["annotations"][0]["frame_index"] == 0


def test_generated_loader_emits_frame_category_datapoints_by_default(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "sam3_finetune"
    write_tiny_dataset(dataset_dir)
    output_json = dataset_dir / "sam3_video_train.json"
    output_json.write_text(
        json.dumps(build_sam3_json(dataset_dir, show_progress=False)),
        encoding="utf-8",
    )

    loader = SAM3_VIDEO_TRAIN_API_FROM_JSON(str(output_json), convert_rle=False)

    assert loader.getDatapointIds() == [0, 1, 2, 3]
    assert pickle.loads(pickle.dumps(loader)).getDatapointIds() == [0, 1, 2, 3]
    images = loader.loadImagesFromDatapoint(0)
    queries, annotations = loader.loadQueriesAndAnnotationsFromDatapoint(0)

    assert [image["file_name"] for image in images] == ["JPEGImages/seq1/000000.jpg"]
    assert Counter(query["query_processing_order"] for query in queries) == {0: 1}
    assert all(query["query_text"] == "pay" for query in queries)
    assert all(query["object_ids_output"] for query in queries)
    assert annotations[0]["bbox"] == [1 / 8, 1 / 6, 3 / 8, 3 / 6]
    assert annotations[0]["object_id"] == 1

    images = loader.loadImagesFromDatapoint(1)
    queries, annotations = loader.loadQueriesAndAnnotationsFromDatapoint(1)

    assert [image["file_name"] for image in images] == [
        "JPEGImages/seq1/000000.jpg",
    ]
    assert Counter(query["query_processing_order"] for query in queries) == {0: 1}
    assert [query["query_text"] for query in queries] == ["pickup"]
    assert queries[0]["object_ids_output"] == []
    assert annotations == []

    images = loader.loadImagesFromDatapoint(2)
    queries, annotations = loader.loadQueriesAndAnnotationsFromDatapoint(2)

    assert [image["file_name"] for image in images] == [
        "JPEGImages/seq1/000001.jpg",
    ]
    assert Counter(query["query_processing_order"] for query in queries) == {0: 1}
    assert all(query["query_text"] == "pay" for query in queries)
    assert annotations[0]["object_id"] == 1

    images = loader.loadImagesFromDatapoint(3)
    queries, annotations = loader.loadQueriesAndAnnotationsFromDatapoint(3)

    assert [image["file_name"] for image in images] == [
        "JPEGImages/seq1/000001.jpg",
    ]
    assert all(query["query_text"] == "pickup" for query in queries)
    assert all(query["object_ids_output"] for query in queries)
    assert annotations[0]["object_id"] == 2


def test_generated_loader_keeps_video_object_mode(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "sam3_finetune"
    write_tiny_dataset(dataset_dir)
    output_json = dataset_dir / "sam3_video_train.json"
    output_json.write_text(
        json.dumps(
            build_sam3_json(
                dataset_dir,
                show_progress=False,
                sample_unit="video_object",
            )
        ),
        encoding="utf-8",
    )

    loader = SAM3_VIDEO_TRAIN_API_FROM_JSON(str(output_json), convert_rle=False)

    assert loader.getDatapointIds() == [0, 1]
    images = loader.loadImagesFromDatapoint(0)
    queries, annotations = loader.loadQueriesAndAnnotationsFromDatapoint(0)

    assert [image["file_name"] for image in images] == [
        "JPEGImages/seq1/000000.jpg",
        "JPEGImages/seq1/000001.jpg",
    ]
    assert Counter(query["query_processing_order"] for query in queries) == {
        0: 1,
        1: 1,
    }
    assert all(query["query_text"] == "pay" for query in queries)
    assert annotations[0]["object_id"] == 1


def test_convert_dataset_writes_json_and_yaml(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "sam3_finetune"
    write_tiny_dataset(dataset_dir)

    summary = convert_dataset(
        ConvertOptions(
            dataset_dir=dataset_dir,
            output_json=dataset_dir / "sam3_video_train.json",
            output_yaml=dataset_dir / "pokeworks_cashier_ft.yaml",
            overwrite=False,
            yaml_only=False,
            dry_run=False,
            experiment_log_dir="",
            bpe_path="${oc.env:SAM3_BPE_PATH,}",
            resolution=512,
            max_epochs=3,
            train_batch_size=1,
            val_batch_size=1,
            num_workers=0,
            num_stages_sample=1,
            stage_stride_min=1,
            stage_stride_max=1,
            max_masklet_num_in_video=20,
            category_chunk_size=None,
            max_frames=None,
            dataset_name="pokeworks_cashier",
            sample_unit="frame_category",
            convert_workers=1,
            show_progress=False,
        )
    )

    assert summary["videos"] == 1
    assert summary["annotations"] == 3
    assert summary["samples"] == 4
    assert summary["positive_samples"] == 3
    assert summary["negative_samples"] == 1
    assert summary["sample_unit"] == "frame_category"
    assert summary["convert_workers"] == 1
    yaml_text = (dataset_dir / "pokeworks_cashier_ft.yaml").read_text(
        encoding="utf-8"
    )
    assert "tools.sam3_finetune_dataset_to_sam3_json.SAM3_TRAIN_API_FROM_JSON" in yaml_text
    assert " -c configs/pal/pokeworks_cashier_ft.yaml " in yaml_text
    assert " -c /" not in yaml_text
    assert "sample_unit: \"frame_category\"" in yaml_text
    assert "_target_: sam3.train.data.sam3_image_dataset.Sam3ImageDataset" in yaml_text
    assert "_target_: sam3.train.data.sam3_video_dataset.VideoGroundingDataset" not in yaml_text
    assert "pal_video_val: null" in yaml_text
    assert "pal_video_train: ${pal_video_train.loss}" in yaml_text
    assert "all: ${pal_video_train.loss}" not in yaml_text
    assert "mode: train_only" in yaml_text
    assert "persistent_workers" not in yaml_text
    assert "prefetch_factor" not in yaml_text
    assert "find_unused_parameters: false" in yaml_text
    assert "static_graph: true" in yaml_text
    assert f"_target_: {MODEL_TARGET}" in yaml_text
    assert "_target_: sam3.model_builder.build_sam3_image_model" not in yaml_text
    assert "_target_: sam3.model_builder.build_sam3_video_model" not in yaml_text
    assert "eval_mode: false" in yaml_text
    assert "detector.backbone" not in yaml_text
    assert "backbone.vision_backbone.*" in yaml_text
    assert "backbone.language_backbone.*" in yaml_text
    assert "apply_to: \"backbone.vision_backbone.trunk\"" in yaml_text
    assert "square:" not in yaml_text
    assert "ratio: 1.0" in yaml_text
    assert "bottom_right: true" in yaml_text
    assert "consistent_transform: false" in yaml_text
    max_size_block = re.search(
        r"_target_: sam3\.train\.transforms\.basic\.get_random_resize_max_size"
        r".*?consistent_transform:",
        yaml_text,
        re.DOTALL,
    )
    assert max_size_block is not None
    assert "resolution: 512" in yaml_text
    assert "num_stages_sample: 1" in yaml_text


def test_convert_dataset_video_object_mode_keeps_video_dataset_yaml(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "sam3_finetune"
    write_tiny_dataset(dataset_dir)

    summary = convert_dataset(
        ConvertOptions(
            dataset_dir=dataset_dir,
            output_json=dataset_dir / "sam3_video_train.json",
            output_yaml=dataset_dir / "pokeworks_cashier_ft.yaml",
            overwrite=False,
            yaml_only=False,
            dry_run=False,
            experiment_log_dir="",
            bpe_path="${oc.env:SAM3_BPE_PATH,}",
            resolution=512,
            max_epochs=3,
            train_batch_size=1,
            val_batch_size=1,
            num_workers=0,
            num_stages_sample=1,
            stage_stride_min=1,
            stage_stride_max=1,
            max_masklet_num_in_video=20,
            category_chunk_size=None,
            max_frames=None,
            dataset_name="pokeworks_cashier",
            sample_unit="video_object",
            convert_workers=1,
            show_progress=False,
        )
    )

    assert summary["sample_unit"] == "video_object"
    assert summary["samples"] == 2
    yaml_text = (dataset_dir / "pokeworks_cashier_ft.yaml").read_text(
        encoding="utf-8"
    )
    assert "sample_unit: \"video_object\"" in yaml_text
    assert "_target_: sam3.train.data.sam3_video_dataset.VideoGroundingDataset" in yaml_text
    assert "_target_: sam3.train.data.sam3_image_dataset.Sam3ImageDataset" not in yaml_text
    assert "max_masklet_num_in_video: ${scratch.max_masklet_num_in_video}" in yaml_text
