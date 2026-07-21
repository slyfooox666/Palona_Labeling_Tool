from __future__ import annotations

import os
import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from vision_pipeline.models.qwen3_vl import Qwen3VLClient, Qwen3VLConfig
from vision_pipeline.models.qwen3_vl.client import _parse_bbox, _parse_json_content


def test_qwen3_vl_config_resolves_endpoint_env() -> None:
    original = os.environ.get("TEST_QWEN3_VL_ENDPOINT")
    os.environ["TEST_QWEN3_VL_ENDPOINT"] = "http://localhost:8000"
    try:
        config = Qwen3VLConfig.from_mapping(
            {
                "runtime": {
                    "endpoint_env": "TEST_QWEN3_VL_ENDPOINT",
                    "model_name": "Qwen/Qwen3-VL-8B-Instruct",
                }
            }
        )
    finally:
        if original is None:
            os.environ.pop("TEST_QWEN3_VL_ENDPOINT", None)
        else:
            os.environ["TEST_QWEN3_VL_ENDPOINT"] = original

    assert config.endpoint == "http://localhost:8000"
    assert config.default_target == "person"


def test_qwen3_vl_config_ignores_empty_endpoint_when_env_is_set() -> None:
    original = os.environ.get("TEST_QWEN3_VL_ENDPOINT")
    os.environ["TEST_QWEN3_VL_ENDPOINT"] = "http://localhost:8080"
    try:
        config = Qwen3VLConfig.from_mapping(
            {
                "runtime": {
                    "endpoint": "",
                    "endpoint_env": "TEST_QWEN3_VL_ENDPOINT",
                }
            }
        )
    finally:
        if original is None:
            os.environ.pop("TEST_QWEN3_VL_ENDPOINT", None)
        else:
            os.environ["TEST_QWEN3_VL_ENDPOINT"] = original

    assert config.endpoint == "http://localhost:8080"


def test_qwen3_vl_json_parser_handles_markdown_fences() -> None:
    parsed = _parse_json_content(
        '```json\n{"detections":[{"label":"person","confidence":0.9}]}\n```'
    )

    assert parsed == {"detections": [{"label": "person", "confidence": 0.9}]}


def test_qwen3_vl_json_parser_recovers_truncated_detection_list() -> None:
    parsed = _parse_json_content(
        '{"detections":['
        '{"label":"person","confidence":0.99,"bbox_xyxy":[107,472,200,828]},'
        '{"label":"person","confidence":0.98,"bbox_xyxy":[147,456,240,828]},'
        '{"label":"person","confidence":0.97,"bbox_xyxy":[380,328,480,5'
    )

    assert isinstance(parsed, dict)
    assert parsed["parse_status"] == "partial_recovery"
    assert parsed["detections"] == [
        {"label": "person", "confidence": 0.99, "bbox_xyxy": [107, 472, 200, 828]},
        {"label": "person", "confidence": 0.98, "bbox_xyxy": [147, 456, 240, 828]},
    ]


def test_qwen3_vl_detection_conversion() -> None:
    client = Qwen3VLClient(
        Qwen3VLConfig(
            model_id="qwen3_test",
            endpoint="http://localhost:8000",
            model_name="Qwen/Qwen3-VL-8B-Instruct",
        )
    )

    detections = client._detections_from_result(
        {
            "detections": [
                {
                    "label": "person",
                    "confidence": 86.0,
                    "bbox_xyxy": [10, 20, 110, 220],
                }
            ]
        },
        target="person",
        timestamp_seconds=1.5,
        image_size=(200, 300),
        bbox_coordinate_format="absolute",
    )

    assert len(detections) == 1
    assert detections[0].label == "person"
    assert detections[0].confidence == 0.86
    assert detections[0].bbox.as_xyxy() == (10.0, 20.0, 110.0, 220.0)
    assert detections[0].metadata["timestamp_seconds"] == 1.5


def test_qwen3_vl_bbox_parser_supports_qwen1000_coordinates() -> None:
    bbox = _parse_bbox(
        [100, 200, 300, 400],
        image_size=(2000, 1000),
        coordinate_format="qwen1000",
    )

    assert bbox is not None
    assert bbox.as_xyxy() == (200.0, 200.0, 600.0, 400.0)
