from __future__ import annotations

import json
import sys
from pathlib import Path


TOOLS_ROOT = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS_ROOT))

from post_train import (  # noqa: E402
    ToolError,
    checkpoint_path_from_download_result,
    detector_key_for_finetune_key,
    find_default_train_checkpoint,
    infer_prompts_from_sam3_json,
    merge_detector_weights,
    render_model_config,
)


class FakeTensor:
    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape


def test_detector_key_for_finetune_key_prefixes_image_model_keys() -> None:
    assert detector_key_for_finetune_key("backbone.weight") == "detector.backbone.weight"
    assert (
        detector_key_for_finetune_key("module.backbone.bias")
        == "detector.backbone.bias"
    )
    assert (
        detector_key_for_finetune_key("detector.backbone.weight")
        == "detector.backbone.weight"
    )


def test_merge_detector_weights_updates_existing_matching_keys() -> None:
    base = {
        "detector.backbone.weight": FakeTensor((2, 2)),
        "detector.head.bias": FakeTensor((2,)),
        "tracker.memory.weight": FakeTensor((4,)),
    }
    finetune = {
        "backbone.weight": FakeTensor((2, 2)),
        "module.head.bias": FakeTensor((2,)),
        "unknown.weight": FakeTensor((1,)),
    }

    merged, summary = merge_detector_weights(base, finetune)

    assert merged["detector.backbone.weight"] is finetune["backbone.weight"]
    assert merged["detector.head.bias"] is finetune["module.head.bias"]
    assert merged["tracker.memory.weight"] is base["tracker.memory.weight"]
    assert "detector.unknown.weight" not in merged
    assert summary.updated == 2
    assert summary.skipped_missing == 1
    assert summary.skipped_shape == 0


def test_merge_detector_weights_skips_shape_mismatches() -> None:
    base = {"detector.backbone.weight": FakeTensor((2, 2))}
    finetune = {"backbone.weight": FakeTensor((3, 2))}

    merged, summary = merge_detector_weights(base, finetune)

    assert merged["detector.backbone.weight"] is base["detector.backbone.weight"]
    assert summary.updated == 0
    assert summary.skipped_shape == 1
    assert "detector.backbone.weight" in summary.shape_examples[0]


def test_render_model_config_contains_checkpoint_and_prompts(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "sam3_video_ft.pt"

    text = render_model_config(
        model_id="pokeworks_cashier_sam3_video_ft",
        checkpoint_path=checkpoint,
        prompts=["cashier machine", "person"],
        native_version="sam3",
    )

    assert 'model_id: "pokeworks_cashier_sam3_video_ft"' in text
    assert f"checkpoint_path: {json.dumps(str(checkpoint))}" in text
    assert '    - "cashier machine"' in text
    assert '    - "person"' in text
    assert "adapter: vision_pipeline.models.sam3.SAM3ModelAdapter" in text


def test_infer_prompts_from_sam3_json_reads_categories(tmp_path: Path) -> None:
    sam3_json = tmp_path / "sam3_video_train.json"
    sam3_json.write_text(
        json.dumps(
            {
                "categories": [
                    {"id": 1, "name": "cashier machine"},
                    {"id": 2, "name": "person"},
                    {"id": 3, "name": "cashier machine"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert infer_prompts_from_sam3_json(sam3_json) == ["cashier machine", "person"]


def test_find_default_train_checkpoint_requires_single_match(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "sam3_finetune"
    checkpoint = dataset_dir / "logs" / "run_a" / "checkpoints" / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("placeholder", encoding="utf-8")

    assert find_default_train_checkpoint(dataset_dir) == checkpoint

    second = dataset_dir / "logs" / "run_b" / "checkpoints" / "checkpoint.pt"
    second.parent.mkdir(parents=True)
    second.write_text("placeholder", encoding="utf-8")

    try:
        find_default_train_checkpoint(dataset_dir)
    except ToolError as exc:
        assert "multiple" in str(exc).lower()
    else:
        raise AssertionError("expected ToolError for multiple checkpoint matches")


def test_checkpoint_path_from_download_result_accepts_nested_values() -> None:
    path = checkpoint_path_from_download_result(
        {"metadata": "x", "checkpoint_path": "/tmp/sam3.pt"}
    )
    assert path == Path("/tmp/sam3.pt")

    nested = checkpoint_path_from_download_result(("ignored", {"path": "/tmp/base.pt"}))
    assert nested == Path("/tmp/base.pt")
