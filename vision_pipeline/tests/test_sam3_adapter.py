from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from vision_pipeline.models.sam3 import SAM3Config, SAM3ModelAdapter


def test_sam3_config_reads_prompts_and_runtime() -> None:
    config = SAM3Config.from_mapping(
        {
            "model_id": "sam3_test",
            "runtime": {
                "model_name": "facebook/sam3",
                "device": "cpu",
                "device_map": "",
                "dtype": "float32",
                "native_version": "sam3.1",
                "checkpoint_path": "/models/sam3.pt",
                "compile_model": True,
                "gpus_to_use": "0,1",
            },
            "inference": {
                "prompts": ["person", "employee"],
                "confidence_threshold": 0.7,
                "max_frame_num_to_track": 12,
            },
        }
    )

    assert config.model_id == "sam3_test"
    assert config.device == "cpu"
    assert config.device_map is None
    assert config.native_version == "sam3.1"
    assert config.checkpoint_path == "/models/sam3.pt"
    assert config.compile_model is True
    assert config.gpus_to_use == (0, 1)
    assert config.prompts == ("person", "employee")
    assert config.confidence_threshold == 0.7
    assert config.max_frame_num_to_track == 12


def test_sam3_image_results_convert_to_detections() -> None:
    adapter = SAM3ModelAdapter(
        SAM3Config(
            model_id="sam3_test",
            confidence_threshold=0.5,
        )
    )

    detections = adapter._detections_from_result(
        {
            "boxes": [[1, 2, 11, 22], [20, 30, 40, 50]],
            "scores": [0.9, 0.2],
        },
        label="person",
        timestamp_seconds=3.5,
    )

    assert len(detections) == 1
    assert detections[0].label == "person"
    assert detections[0].confidence == 0.9
    assert detections[0].bbox.as_xyxy() == (1.0, 2.0, 11.0, 22.0)
    assert detections[0].metadata["timestamp_seconds"] == 3.5


def test_sam3_video_results_convert_to_tracks_with_prompt_labels() -> None:
    adapter = SAM3ModelAdapter(
        SAM3Config(
            model_id="sam3_test",
            confidence_threshold=0.5,
        )
    )

    tracks = adapter._tracks_from_result(
        {
            "boxes": [[1, 2, 11, 22], [20, 30, 40, 50]],
            "scores": [0.9, 0.8],
            "object_ids": [7, 8],
            "prompt_to_obj_ids": {
                "person": [7],
                "employee": [8],
            },
        },
        timestamp_seconds=4.0,
        frame_index=2,
        default_label="sam3_object",
    )

    assert [track.track_id for track in tracks] == ["7", "8"]
    assert [track.label for track in tracks] == ["person", "employee"]
    assert tracks[0].metadata["frame_index"] == 2


def test_sam3_prompt_resolution_splits_comma_separated_concepts() -> None:
    adapter = SAM3ModelAdapter(SAM3Config(model_id="sam3_test"))

    prompts = adapter._resolve_prompts(
        ["person, soap bottle", "water faucet", "paper towel"]
    )

    assert prompts == ("person", "soap bottle", "water faucet", "paper towel")


def test_sam3_native_video_adds_each_prompt_separately() -> None:
    class FakePredictor:
        def __init__(self) -> None:
            self.prompts = []
            self.started_sessions = []
            self.closed_sessions = []

        def handle_request(self, request):
            if request["type"] == "start_session":
                session_id = f"session-{len(self.started_sessions) + 1}"
                self.started_sessions.append(session_id)
                return {"session_id": session_id}
            if request["type"] == "add_prompt":
                self.prompts.append(request["text"])
                return {
                    "frame_index": request["frame_index"],
                    "outputs": {
                        "out_obj_ids": [1],
                        "out_binary_masks": [
                            [
                                [0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0],
                            ]
                        ],
                    },
                }
            if request["type"] == "close_session":
                self.closed_sessions.append(request["session_id"])
                return {}
            raise AssertionError(f"unexpected request: {request}")

        def handle_stream_request(self, request):
            return iter(())

    adapter = SAM3ModelAdapter(SAM3Config(model_id="sam3_test"))
    predictor = FakePredictor()
    adapter._native_video_predictor = predictor

    frame_results = adapter.track_video_path(
        "video.mp4",
        prompt="person, soap bottle",
        fps=10.0,
    )

    assert predictor.prompts == ["person", "soap bottle"]
    assert predictor.started_sessions == ["session-1", "session-2"]
    assert predictor.closed_sessions == ["session-1", "session-2"]
    assert len(frame_results) == 1
    assert [track.track_id for track in frame_results[0].tracks] == ["p0:1", "p1:1"]
    assert [track.label for track in frame_results[0].tracks] == [
        "person",
        "soap bottle",
    ]


def test_sam3_native_video_output_converts_masks_to_tracks() -> None:
    adapter = SAM3ModelAdapter(
        SAM3Config(
            model_id="sam3_test",
            confidence_threshold=0.5,
        )
    )

    tracks = adapter._tracks_from_native_output(
        {
            "out_obj_ids": [3],
            "out_binary_masks": [
                [
                    [0, 0, 0, 0],
                    [0, 1, 1, 0],
                    [0, 1, 1, 0],
                ]
            ],
            "prompt_to_obj_ids": {
                "basket": [3],
            },
        },
        timestamp_seconds=2.0,
        frame_index=4,
        default_label="sam3_object",
    )

    assert len(tracks) == 1
    assert tracks[0].track_id == "3"
    assert tracks[0].label == "basket"
    assert tracks[0].bbox.as_xyxy() == (1.0, 1.0, 3.0, 3.0)
    assert tracks[0].metadata["mask"]
    assert tracks[0].metadata["video_mode"] == "whole"
