from __future__ import annotations

import sys
from pathlib import Path

TOOLS_ROOT = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS_ROOT))

from build_interaction_contours import (  # noqa: E402
    InteractionConfig,
    build_interaction_contours,
    parse_fix_mappings,
    parse_golden_objects,
    parse_interaction_configs,
)


def make_track(
    track_id: str,
    contour: list[list[int]],
    bbox: list[float],
) -> dict:
    return {
        "track_id": track_id,
        "label": "object",
        "confidence": 1.0,
        "state": "active",
        "bbox_xyxy": bbox,
        "metadata": {},
        "contours_xy": [contour],
        "contours_format": "absolute_xy",
    }


def test_build_interaction_contours_merges_only_configured_interactions() -> None:
    contour_data = {
        "input_type": "video",
        "video": "clip.mp4",
        "frames": [
            {
                "frame_index": 0,
                "timestamp_seconds": 6.9,
                "tracks": [
                    make_track("p0:0", [[0, 0], [1, 0]], [0, 0, 1, 1]),
                    make_track("p2:0", [[10, 10], [11, 10]], [10, 10, 11, 11]),
                ],
            },
            {
                "frame_index": 1,
                "timestamp_seconds": 7.0,
                "tracks": [
                    make_track("p0:0", [[0, 0], [1, 0]], [0, 0, 1, 1]),
                    make_track("p2:0", [[10, 10], [11, 10]], [10, 10, 11, 11]),
                    make_track("unrelated", [[20, 20], [21, 20]], [20, 20, 21, 21]),
                ],
            },
            {
                "frame_index": 2,
                "timestamp_seconds": 38.0,
                "tracks": [
                    make_track("p0:0", [[2, 2], [3, 2]], [2, 2, 3, 3]),
                    make_track("p2:0", [[12, 12], [13, 12]], [12, 12, 13, 13]),
                ],
            },
            {
                "frame_index": 3,
                "timestamp_seconds": 38.1,
                "tracks": [
                    make_track("p0:0", [[0, 0], [1, 0]], [0, 0, 1, 1]),
                    make_track("p2:0", [[10, 10], [11, 10]], [10, 10, 11, 11]),
                ],
            },
        ],
    }
    interactions = [
        InteractionConfig(
            track_id="i0",
            label="wash hand",
            confidence=1.0,
            start_time=7.0,
            end_time=38.0,
            track_id_list=["p0:0", "p2:0"],
        )
    ]

    output = build_interaction_contours(contour_data, interactions)

    assert [len(frame["tracks"]) for frame in output["frames"]] == [0, 1, 1, 0]
    interaction_track = output["frames"][1]["tracks"][0]
    assert interaction_track["track_id"] == "i0"
    assert interaction_track["label"] == "wash hand"
    assert interaction_track["bbox_xyxy"] == [0.0, 0.0, 11.0, 11.0]
    assert interaction_track["contours_xy"] == [
        [[0, 0], [1, 0]],
        [[10, 10], [11, 10]],
    ]
    assert all(
        track["track_id"].startswith("i")
        for frame in output["frames"]
        for track in frame["tracks"]
    )


def test_build_interaction_contours_skips_missing_source_track() -> None:
    contour_data = {
        "frames": [
            {
                "frame_index": 0,
                "timestamp_seconds": 10.0,
                "tracks": [make_track("p0:0", [[0, 0], [1, 0]], [0, 0, 1, 1])],
            }
        ]
    }
    interactions = [
        InteractionConfig(
            track_id="i0",
            label="wash hand",
            confidence=1.0,
            start_time=7.0,
            end_time=38.0,
            track_id_list=["p0:0", "p2:0"],
        )
    ]

    output = build_interaction_contours(contour_data, interactions)

    assert output["frames"][0]["tracks"] == []


def test_build_interaction_contours_uses_golden_object_during_interaction() -> None:
    contour_data = {
        "frames": [
            {
                "frame_index": 0,
                "timestamp_seconds": 6.9,
                "tracks": [
                    make_track("p0:0", [[0, 0], [1, 0], [1, 1]], [0, 0, 1, 1])
                ],
            },
            {
                "frame_index": 1,
                "timestamp_seconds": 7.0,
                "tracks": [
                    make_track("p0:0", [[0, 0], [1, 0], [1, 1]], [0, 0, 1, 1])
                ],
            },
            {
                "frame_index": 2,
                "timestamp_seconds": 38.0,
                "tracks": [
                    make_track("p0:0", [[2, 2], [3, 2], [3, 3]], [2, 2, 3, 3])
                ],
            },
            {
                "frame_index": 3,
                "timestamp_seconds": 38.1,
                "tracks": [
                    make_track("p0:0", [[4, 4], [5, 4], [5, 5]], [4, 4, 5, 5])
                ],
            },
        ]
    }
    interactions = [
        InteractionConfig(
            track_id="i0",
            label="at cashier",
            confidence=1.0,
            start_time=7.0,
            end_time=38.0,
            track_id_list=["p0:0", "p:pokeworks-cashier"],
        )
    ]
    golden_objects = parse_golden_objects(
        [
            {
                "track_id": "p:pokeworks-cashier",
                "contours_xy": [
                    [[10, 10], [11, 10], [11, 11]],
                    [[20, 20], [21, 20], [21, 21]],
                ],
            }
        ]
    )

    output = build_interaction_contours(
        contour_data,
        interactions,
        golden_objects=golden_objects,
    )

    assert [len(frame["tracks"]) for frame in output["frames"]] == [0, 1, 1, 0]
    interaction_track = output["frames"][1]["tracks"][0]
    assert interaction_track["contours_xy"] == [
        [[0, 0], [1, 0], [1, 1]],
        [[10, 10], [11, 10], [11, 11]],
        [[20, 20], [21, 20], [21, 21]],
    ]
    assert interaction_track["bbox_xyxy"] == [0.0, 0.0, 21.0, 21.0]


def test_build_interaction_contours_uses_fix_only_when_requested_id_missing() -> None:
    contour_data = {
        "frames": [
            {
                "frame_index": 0,
                "timestamp_seconds": 7.0,
                "tracks": [
                    make_track("p0:0", [[0, 0], [1, 0], [1, 1]], [0, 0, 1, 1]),
                    make_track("p1:0", [[5, 5], [6, 5], [6, 6]], [5, 5, 6, 6]),
                ],
            },
            {
                "frame_index": 1,
                "timestamp_seconds": 8.0,
                "tracks": [
                    make_track("p0:0", [[2, 2], [3, 2], [3, 3]], [2, 2, 3, 3])
                ],
            },
        ]
    }
    interactions = [
        InteractionConfig(
            track_id="i0",
            label="at cashier",
            confidence=1.0,
            start_time=7.0,
            end_time=8.0,
            track_id_list=["p0:0", "p1:0"],
        )
    ]
    golden_objects = parse_golden_objects(
        [
            {
                "track_id": "p:pokeworks-cashier",
                "contours_xy": [[[10, 10], [11, 10], [11, 11]]],
            }
        ]
    )

    output = build_interaction_contours(
        contour_data,
        interactions,
        golden_objects=golden_objects,
        fix_mappings={"p1:0": "p:pokeworks-cashier"},
    )

    assert output["frames"][0]["tracks"][0]["contours_xy"] == [
        [[0, 0], [1, 0], [1, 1]],
        [[5, 5], [6, 5], [6, 6]],
    ]
    assert output["frames"][1]["tracks"][0]["contours_xy"] == [
        [[2, 2], [3, 2], [3, 3]],
        [[10, 10], [11, 10], [11, 11]],
    ]


def test_parse_golden_objects_accepts_list_shape_and_fills_defaults() -> None:
    golden_objects = parse_golden_objects(
        [
            {
                "track_id": "p:pokeworks-cashier",
                "contours_xy": [[[10, 10], [11, 10], [11, 11]]],
            }
        ]
    )

    golden_object = golden_objects["p:pokeworks-cashier"]
    assert golden_object["label"] == "p:pokeworks-cashier"
    assert golden_object["confidence"] == 1.0
    assert golden_object["state"] == "active"
    assert golden_object["contours_format"] == "absolute_xy"
    assert golden_object["bbox_xyxy"] == [10.0, 10.0, 11.0, 11.0]


def test_parse_fix_mappings_accepts_comma_delimited_values() -> None:
    assert parse_fix_mappings(
        ["p1:0==p:pokeworks-cashier,p2:0==p:backup", "p3:0==p:other"]
    ) == {
        "p1:0": "p:pokeworks-cashier",
        "p2:0": "p:backup",
        "p3:0": "p:other",
    }


def test_parse_interaction_configs_accepts_top_level_list_or_object() -> None:
    record = {
        "track_id": "i0",
        "label": "wash hand",
        "confidence": 1.0,
        "start_time": 7.0,
        "end_time": 38.0,
        "track_id_list": ["p0:0", "p2:0"],
    }

    assert parse_interaction_configs([record])[0].track_id == "i0"
    assert parse_interaction_configs({"interactions": [record]})[0].label == "wash hand"
