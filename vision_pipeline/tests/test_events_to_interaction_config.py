from __future__ import annotations

import json
import sys
from pathlib import Path

TOOLS_ROOT = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS_ROOT))

from events_to_interaction_config import (  # noqa: E402
    convert_file,
    events_to_interaction_config,
    output_path_for_events_file,
)


def test_events_to_interaction_config_batch_shape_skips_false_events() -> None:
    data = [
        {
            "video": "clip.mp4",
            "result": {
                "event_list": [
                    {
                        "event": "wait",
                        "person_id": "p0:4",
                        "cashier_machine_id": "p1:0",
                        "start_time": 0,
                        "end_time": 12,
                    },
                    {
                        "event": "bad",
                        "person_id": "p0:9",
                        "cashier_machine_id": "p1:0",
                        "start_time": 1,
                        "end_time": 2,
                        "false_positive": True,
                    },
                    {
                        "event": "bad2",
                        "person_id": "p0:8",
                        "cashier_machine_id": "p1:0",
                        "start_time": 3,
                        "end_time": 4,
                        "false_position": "true",
                    },
                    {
                        "event": "operate",
                        "person_id": "p0:7",
                        "cashier_machine_id": "p1:0",
                        "start_time": 4,
                        "end_time": 13,
                    },
                ]
            },
        }
    ]

    interactions = events_to_interaction_config(data)

    assert [interaction["track_id"] for interaction in interactions] == ["i1", "i2"]
    assert interactions[0]["label"] == "wait"
    assert interactions[0]["confidence"] == 1.0
    assert interactions[0]["start_time"] == 0.0
    assert interactions[0]["end_time"] == 12.0
    assert interactions[0]["track_id_list"] == ["p0:4", "p1:0"]
    assert interactions[1]["label"] == "operate"
    assert interactions[1]["track_id_list"] == ["p0:7", "p1:0"]


def test_events_to_interaction_config_reuses_id_for_same_id_set() -> None:
    data = {
        "event_list": [
            {
                "event": "wait",
                "person_id": "p0:4",
                "cashier_machine_id": "p1:0",
                "start_time": 0,
                "end_time": 10,
            },
            {
                "event": "operate",
                "cashier_machine_id": "p1:0",
                "person_id": "p0:4",
                "start_time": 12,
                "end_time": 20,
            },
            {
                "event": "pay",
                "person_id": "p0:7",
                "cashier_machine_id": "p1:0",
                "start_time": 21,
                "end_time": 25,
            },
        ]
    }

    interactions = events_to_interaction_config(data)

    assert [interaction["track_id"] for interaction in interactions] == [
        "i1",
        "i1",
        "i2",
    ]
    assert interactions[0]["track_id_list"] == ["p0:4", "p1:0"]
    assert interactions[1]["track_id_list"] == ["p1:0", "p0:4"]


def test_output_path_for_events_file_replaces_suffix() -> None:
    path = Path("agent/vision/clip_events.json")

    assert output_path_for_events_file(path) == Path(
        "agent/vision/clip_interaction_config.json"
    )


def test_convert_file_writes_interaction_config(tmp_path: Path) -> None:
    input_path = tmp_path / "clip_events.json"
    input_path.write_text(
        json.dumps(
            {
                "event_list": [
                    {
                        "event": "pay",
                        "person_id": "p0:0",
                        "cashier_machine_id": "p1:0",
                        "start_time": 2,
                        "end_time": 5,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    output_path, count = convert_file(input_path)

    assert count == 1
    assert output_path == tmp_path / "clip_interaction_config.json"
    assert json.loads(output_path.read_text(encoding="utf-8")) == [
        {
            "track_id": "i1",
            "label": "pay",
            "confidence": 1.0,
            "start_time": 2.0,
            "end_time": 5.0,
            "track_id_list": ["p0:0", "p1:0"],
        }
    ]
