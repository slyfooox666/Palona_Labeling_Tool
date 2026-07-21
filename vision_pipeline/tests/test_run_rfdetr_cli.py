from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(SRC_ROOT))

from vision_pipeline.core.schemas import BoundingBox, Track

from vp_cli import (
    LatencyTimer,
    Sam3VideoChunk,
    TrackedFrameRecord,
    apply_source_frame_timestamps,
    annotated_variant_path,
    build_sam3_video_chunk_specs,
    build_continuous_track_appearances,
    build_single_track_appearances,
    contour_json_output_path,
    contour_json_requested,
    detect_input_type,
    emit_json_result,
    detection_classes,
    escape_ffconcat_path,
    frame_results_by_video_frame,
    format_wall_clock_timestamp,
    draw_video_output_fps,
    infer_frame_sequence_fps,
    offset_bbox,
    open_ffmpeg_video_writer,
    parse_roi,
    parse_source_video_start_datetime,
    prepare_sam3_processing_video,
    probe_video_frame_timestamps,
    sam3_tracked_clip_frame_record,
    serialized_contour_annotations_by_frame,
    merge_sam3_chunk_results,
    sampled_draw_video_frame_indexes,
    serialize_tracks,
    serialize_model_metadata,
    split_sam3_video_chunks,
    track_and_merge_sam3_video_chunks,
    tracked_object_clip_output_path,
)
from vision_pipeline.models.sam3.adapter import Sam3FrameResult
from vision_pipeline.utils.video import SampledFrame


def test_detect_input_type_supports_images_and_videos() -> None:
    assert detect_input_type(Path("frame.jpg")) == "image"
    assert detect_input_type(Path("frame.png")) == "image"
    assert detect_input_type(Path("clip.mp4")) == "video"
    assert detect_input_type(Path("clip.mov")) == "video"


class Args:
    classes = None
    tracker = "bot-sort"
    track_classes = ["person"]


def test_detection_classes_default_to_track_classes_when_tracking() -> None:
    assert detection_classes(Args()) == ["person"]


def test_latency_timer_counts_average_frames() -> None:
    timer = LatencyTimer(enabled=True)
    timer.add_duration(0.25, frame_count=5)
    timer.add_duration(0.15, frame_count=3)

    assert timer.frame_count == 8
    assert round(timer.total_seconds, 2) == 0.4


def test_latency_timer_ignores_disabled_measurements() -> None:
    timer = LatencyTimer(enabled=False)
    timer.add_duration(0.25, frame_count=5)

    assert timer.frame_count == 0
    assert timer.total_seconds == 0.0


def test_parse_roi_accepts_comma_or_space_separated_values() -> None:
    assert parse_roi("10,20,110,220") == (10, 20, 110, 220)
    assert parse_roi("10 20 110 220") == (10, 20, 110, 220)


def test_offset_bbox_maps_crop_coordinates_to_full_frame() -> None:
    bbox = offset_bbox(BoundingBox(1, 2, 11, 22), x_offset=100, y_offset=200)

    assert bbox.as_xyxy() == (101, 202, 111, 222)


def test_annotated_variant_path_does_not_overwrite_source() -> None:
    assert annotated_variant_path(Path("frame_roi.jpg")) == Path("frame_roi_boxes.jpg")


def make_track(track_id: str, timestamp_seconds: float, state: str = "active") -> Track:
    return Track(
        track_id=track_id,
        bbox=BoundingBox(0, 0, 10, 20),
        label="person",
        confidence=0.9,
        timestamp_seconds=timestamp_seconds,
        state=state,  # type: ignore[arg-type]
    )


def test_continuous_track_appearances_split_after_missing_frames() -> None:
    records = [
        TrackedFrameRecord(Path("frame0.jpg"), 0.0, [make_track("1", 0.0)]),
        TrackedFrameRecord(Path("frame1.jpg"), 1.0, [make_track("1", 1.0)]),
        TrackedFrameRecord(Path("frame2.jpg"), 2.0, []),
        TrackedFrameRecord(Path("frame3.jpg"), 3.0, [make_track("1", 3.0, "lost")]),
        TrackedFrameRecord(Path("frame4.jpg"), 4.0, [make_track("1", 4.0)]),
    ]

    appearances = build_continuous_track_appearances(records)

    assert [(item.track_id, item.appearance_index) for item in appearances] == [
        ("1", 1),
        ("1", 2),
    ]
    assert [len(item.observations) for item in appearances] == [2, 1]


def test_single_track_appearance_includes_hidden_frames_without_boxes() -> None:
    records = [
        TrackedFrameRecord(Path("frame0.jpg"), 0.0, [make_track("1", 0.0)]),
        TrackedFrameRecord(Path("frame1.jpg"), 1.0, []),
        TrackedFrameRecord(Path("frame2.jpg"), 2.0, [make_track("1", 2.0, "lost")]),
        TrackedFrameRecord(Path("frame3.jpg"), 3.0, [make_track("1", 3.0)]),
        TrackedFrameRecord(Path("frame4.jpg"), 4.0, [make_track("1", 4.0, "lost")]),
        TrackedFrameRecord(Path("frame5.jpg"), 5.0, []),
    ]

    appearances = build_single_track_appearances(records)

    assert [(item.track_id, item.appearance_index) for item in appearances] == [
        ("1", 1)
    ]
    assert [observation.timestamp_seconds for observation in appearances[0].observations] == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
    ]
    assert [
        observation.track.track_id if observation.track else None
        for observation in appearances[0].observations
    ] == ["1", None, None, "1", None]


def test_source_filename_timestamp_formats_overlay_time() -> None:
    source_start = parse_source_video_start_datetime(
        Path("2026-06-24_13-20-00.mp4")
    )

    assert format_wall_clock_timestamp(source_start, 65.7) == "2026-06-24 13:21:05"


def test_tracked_object_clip_output_path_includes_track_and_appearance() -> None:
    output_path = tracked_object_clip_output_path(
        Path("2026-06-24_13-20-00.mp4"),
        Path("/tmp/clips"),
        "object 2",
        3,
    )

    assert output_path == Path(
        "/tmp/clips/2026-06-24_13-20-00_track-object-2_appearance-3.mp4"
    )


def test_escape_ffconcat_path_handles_quotes_and_backslashes() -> None:
    assert escape_ffconcat_path(Path("/tmp/a'b\\c.jpg")) == "/tmp/a'\\''b\\\\c.jpg"


def test_open_ffmpeg_video_writer_uses_h264_pipe(monkeypatch, tmp_path) -> None:
    commands = []

    class DummyProcess:
        stdin = None
        stderr = None

    def fake_popen(command, stdin, stderr):
        commands.append(command)
        return DummyProcess()

    monkeypatch.setattr("vp_cli.shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr("vp_cli.subprocess.Popen", fake_popen)

    writer = open_ffmpeg_video_writer(
        tmp_path / "annotated.mp4",
        fps=20.0,
        width=64,
        height=48,
    )

    assert isinstance(writer, DummyProcess)
    command = commands[0]
    assert command[command.index("-pix_fmt") + 1] == "bgr24"
    assert command[command.index("-s:v") + 1] == "64x48"
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-crf") + 1] == "23"
    assert "mp4v" not in command


def test_sampled_draw_video_frame_indexes_follow_target_fps() -> None:
    assert sampled_draw_video_frame_indexes(
        30,
        source_fps=30.0,
        output_fps=6.0,
    ) == {0, 5, 10, 15, 20, 25}


def test_draw_video_output_fps_clamps_to_source_fps() -> None:
    assert draw_video_output_fps(30.0, None) == 30.0
    assert draw_video_output_fps(30.0, 6.0) == 6.0
    assert draw_video_output_fps(30.0, 60.0) == 30.0


def test_frame_results_map_to_processing_video_by_timestamp() -> None:
    results = [
        Sam3FrameResult(
            timestamp_seconds=index / 6.0,
            image_path=Path("chunk.mp4"),
            tracks=[],
            frame_index=index * 5,
        )
        for index in range(6)
    ]

    indexed = frame_results_by_video_frame(
        results,
        video_fps=6.0,
        frame_result_fps=30.0,
    )

    assert list(indexed) == [0, 1, 2, 3, 4, 5]


def test_prepare_sam3_processing_video_creates_one_cfr_source(
    monkeypatch,
    tmp_path,
) -> None:
    commands = []

    class Completed:
        returncode = 0
        stderr = ""
        stdout = ""

    def fake_run(command, capture_output, text):
        commands.append(command)
        return Completed()

    monkeypatch.setattr("vp_cli.shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr("vp_cli.subprocess.run", fake_run)

    output = prepare_sam3_processing_video(
        Path("input.mkv"),
        tmp_path,
        fps=6.0,
        max_frames=120,
    )

    assert output == tmp_path / "input_sam3_6fps.mp4"
    command = commands[0]
    assert command[command.index("-vf") + 1] == (
        "fps=fps=6:round=near:start_time=0,setpts=N/(6*TB)"
    )
    assert command[command.index("-frames:v") + 1] == "120"
    assert command[command.index("-r") + 1] == "6"


def test_split_sam3_video_chunks_does_not_resample_canonical_video(
    monkeypatch,
    tmp_path,
) -> None:
    commands = []

    class Completed:
        returncode = 0
        stderr = ""
        stdout = ""

    def fake_run(command, capture_output, text):
        commands.append(command)
        return Completed()

    monkeypatch.setattr("vp_cli.shutil.which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr("vp_cli.subprocess.run", fake_run)
    chunk = Sam3VideoChunk(
        0,
        tmp_path / "chunk.mp4",
        0.0,
        12.0,
        0,
        0,
        72,
        processing_fps=6.0,
    )

    split_sam3_video_chunks(Path("canonical.mp4"), [chunk], input_fps=6.0)

    command = commands[0]
    assert "-vf" not in command
    assert command[command.index("-frames:v") + 1] == "72"


def test_serialize_model_metadata_summarizes_masks() -> None:
    metadata = serialize_model_metadata(
        {
            "prompt": "person",
            "mask": [[0, 1], [1, 1]],
        }
    )

    assert metadata["prompt"] == "person"
    assert metadata["has_mask"] is True
    assert "mask" not in metadata


def test_serialize_tracks_can_include_roi_offset_contours() -> None:
    try:
        import cv2  # noqa: F401
        import numpy  # noqa: F401
    except ImportError:
        return

    track = Track(
        track_id="1",
        bbox=BoundingBox(11, 21, 13, 23),
        label="paper towel",
        confidence=0.9,
        timestamp_seconds=0.0,
        metadata={
            "mask": [
                [0, 0, 0],
                [0, 1, 1],
                [0, 1, 0],
            ],
            "roi_xyxy": (10, 20, 13, 23),
        },
    )

    serialized = serialize_tracks(
        [track],
        include_contours=True,
        contour_epsilon_px=0,
    )

    contours = serialized[0]["contours_xy"]
    assert contours
    points = [point for contour in contours for point in contour]
    assert min(point[0] for point in points) >= 10
    assert min(point[1] for point in points) >= 20
    assert serialized[0]["contours_format"] == "absolute_xy"
    assert "mask" not in serialized[0]["metadata"]


def test_contour_json_flag_accepts_stdout_or_path() -> None:
    assert not contour_json_requested(Namespace(contour_json=None))
    assert contour_json_requested(Namespace(contour_json="true"))
    assert contour_json_requested(Namespace(contour_json="out/contours.json"))
    assert not contour_json_requested(Namespace(contour_json="false"))

    assert contour_json_output_path(Namespace(contour_json=None)) is None
    assert contour_json_output_path(Namespace(contour_json="true")) is None
    assert contour_json_output_path(Namespace(contour_json="-")) is None
    assert contour_json_output_path(
        Namespace(contour_json="out/contours.json")
    ) == Path("out/contours.json")


def test_emit_json_result_writes_requested_path(tmp_path) -> None:
    output_path = tmp_path / "nested" / "contours.json"

    emit_json_result({"ok": True}, Namespace(contour_json=str(output_path)))

    assert json.loads(output_path.read_text(encoding="utf-8")) == {"ok": True}


def test_serialized_contour_annotations_map_frame_index_or_timestamp() -> None:
    payload = {
        "frames": [
            {
                "frame_index": 3,
                "tracks": [{"track_id": "1", "contours_xy": [[[1, 2], [3, 4]]]}],
            },
            {
                "timestamp_seconds": 1.25,
                "tracks": [{"track_id": "2", "contours_xy": [[[5, 6], [7, 8]]]}],
            },
        ]
    }

    by_frame = serialized_contour_annotations_by_frame(payload, fps=8.0)

    assert by_frame[3][0]["track_id"] == "1"
    assert by_frame[10][0]["track_id"] == "2"


def test_serialized_contour_annotations_accept_result_wrapper() -> None:
    payload = {
        "video": "clip.mp4",
        "result": {
            "frames": [
                {
                    "frame_index": 1,
                    "detections": [
                        {"label": "soap bottle", "contours_xy": [[[1, 2], [3, 4]]]}
                    ],
                }
            ]
        },
    }

    by_frame = serialized_contour_annotations_by_frame(payload, fps=30.0)

    assert by_frame[1][0]["label"] == "soap bottle"


def make_bbox_track(
    track_id: str,
    bbox_xyxy: tuple[float, float, float, float],
    *,
    label: str = "person",
    timestamp_seconds: float = 0.0,
) -> Track:
    return Track(
        track_id=track_id,
        bbox=BoundingBox(*bbox_xyxy),
        label=label,
        confidence=0.9,
        timestamp_seconds=timestamp_seconds,
    )


def frame_result(frame_index: int, tracks: list[Track]) -> Sam3FrameResult:
    return Sam3FrameResult(
        timestamp_seconds=float(frame_index),
        image_path=Path("chunk.mp4"),
        tracks=tracks,
        frame_index=frame_index,
    )


def test_build_sam3_video_chunk_specs_overlap_schedule() -> None:
    chunks = build_sam3_video_chunk_specs(
        Path("input.mp4"),
        Path("/tmp/chunks"),
        duration_seconds=60.0,
        fps=30.0,
        split_seconds=15.0,
        overlap_seconds=5.0,
    )

    assert [chunk.start_seconds for chunk in chunks] == [
        0.0,
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
    ]
    assert [chunk.duration_seconds for chunk in chunks] == [
        15.0,
        15.0,
        15.0,
        15.0,
        15.0,
        10.0,
    ]
    assert [chunk.emit_start_frame_index for chunk in chunks[:3]] == [0, 450, 750]


def test_build_sam3_video_chunk_specs_uses_processing_fps_for_sampled_chunks() -> None:
    chunks = build_sam3_video_chunk_specs(
        Path("input.mp4"),
        Path("/tmp/chunks"),
        duration_seconds=60.0,
        fps=30.0,
        split_seconds=12.0,
        overlap_seconds=2.0,
        processing_fps=6.0,
    )

    assert [chunk.start_seconds for chunk in chunks] == [
        0.0,
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
    ]
    assert [chunk.duration_seconds for chunk in chunks] == [
        12.0,
        12.0,
        12.0,
        12.0,
        12.0,
        10.0,
    ]
    assert [chunk.max_frames for chunk in chunks] == [72, 72, 72, 72, 72, 60]
    assert [chunk.processing_fps for chunk in chunks] == [6.0] * 6
    assert [chunk.emit_start_frame_index for chunk in chunks[:3]] == [0, 360, 660]


def test_build_sam3_video_chunk_specs_uses_source_frame_timestamps() -> None:
    frame_timestamps = [
        0.0,
        0.49,
        0.99,
        1.49,
        1.99,
        2.49,
        2.99,
        3.49,
        3.99,
        4.49,
    ]

    chunks = build_sam3_video_chunk_specs(
        Path("input.mkv"),
        Path("/tmp/chunks"),
        duration_seconds=5.0,
        fps=2.0,
        split_seconds=3.0,
        overlap_seconds=1.0,
        source_frame_timestamps=frame_timestamps,
    )

    assert [chunk.start_frame_index for chunk in chunks] == [0, 5]
    assert [chunk.emit_start_frame_index for chunk in chunks] == [0, 7]
    assert [chunk.max_frames for chunk in chunks] == [7, 5]


def test_apply_source_frame_timestamps_corrects_nominal_fps_drift() -> None:
    results = [
        frame_result(index, [make_bbox_track("p0:0", (0, 0, 10, 10))])
        for index in range(4)
    ]
    source_frame_timestamps = [0.0, 0.033, 0.067, 0.133]

    remapped = apply_source_frame_timestamps(results, source_frame_timestamps)

    assert [result.timestamp_seconds for result in remapped] == source_frame_timestamps
    assert [result.tracks[0].timestamp_seconds for result in remapped] == (
        source_frame_timestamps
    )


def test_probe_video_frame_timestamps_accepts_trailing_csv_field(
    monkeypatch, tmp_path: Path
) -> None:
    class CompletedProcess:
        returncode = 0
        stdout = "10.000000,\n10.033000,\n10.100000,\n"
        stderr = ""

    monkeypatch.setattr("vp_cli.shutil.which", lambda _name: "/usr/bin/ffprobe")
    monkeypatch.setattr(
        "vp_cli.subprocess.run",
        lambda *_args, **_kwargs: CompletedProcess(),
    )

    timestamps = probe_video_frame_timestamps(tmp_path / "input.mkv")

    assert [round(timestamp, 3) for timestamp in timestamps] == [0.0, 0.033, 0.1]


def test_merge_sam3_chunk_results_keeps_track_id_from_overlap() -> None:
    chunks = [
        Sam3VideoChunk(0, Path("chunk0.mp4"), 0.0, 4.0, 0, 0, 4),
        Sam3VideoChunk(1, Path("chunk1.mp4"), 2.0, 4.0, 2, 4, 4),
    ]
    chunk_results = [
        [
            frame_result(index, [make_bbox_track("p0:0", (0, 0, 10, 10))])
            for index in range(4)
        ],
        [
            frame_result(index, [make_bbox_track("p0:0", (1, 1, 11, 11))])
            for index in range(4)
        ],
    ]

    merged = merge_sam3_chunk_results(chunks, chunk_results, fps=1.0)

    assert [result.frame_index for result in merged] == [0, 1, 2, 3, 4, 5]
    assert [result.tracks[0].track_id for result in merged] == ["p0:0"] * 6
    assert [result.tracks[0].bbox.as_xyxy() for result in merged] == [
        (0, 0, 10, 10),
        (0, 0, 10, 10),
        (1, 1, 11, 11),
        (1, 1, 11, 11),
        (1, 1, 11, 11),
        (1, 1, 11, 11),
    ]


def test_merge_sam3_chunk_results_maps_sampled_chunk_frames_to_source_frames() -> None:
    chunks = [
        Sam3VideoChunk(
            0,
            Path("chunk0.mp4"),
            0.0,
            2.0,
            0,
            0,
            12,
            processing_fps=6.0,
        ),
    ]
    chunk_results = [
        [
            frame_result(index, [make_bbox_track("p0:0", (0, 0, 10, 10))])
            for index in [0, 1, 2]
        ],
    ]

    merged = merge_sam3_chunk_results(chunks, chunk_results, fps=30.0)

    assert [result.frame_index for result in merged] == [0, 5, 10]
    assert [round(result.timestamp_seconds, 3) for result in merged] == [
        0.0,
        0.167,
        0.333,
    ]


def test_merge_sam3_chunk_results_uses_actual_chunk_start_frame() -> None:
    chunk = Sam3VideoChunk(
        28,
        Path("chunk28.mp4"),
        56.0,
        3.0,
        1681,
        1711,
        90,
    )
    chunk_results = [[frame_result(index, []) for index in [0, 39, 40]]]

    merged = merge_sam3_chunk_results([chunk], chunk_results, fps=30.0)

    assert [result.frame_index for result in merged] == [1720, 1721]
    assert [round(result.timestamp_seconds, 3) for result in merged] == [
        57.333,
        57.367,
    ]


def test_track_sam3_video_chunks_reuses_predictor_between_chunks() -> None:
    class DummyAdapter:
        def __init__(self) -> None:
            self.calls = []

        def track_video_path(self, path, prompt, fps, max_frames):
            self.calls.append(
                {
                    "path": path,
                    "prompt": prompt,
                    "fps": fps,
                    "max_frames": max_frames,
                }
            )
            return [frame_result(0, [])]

    chunks = [
        Sam3VideoChunk(
            0,
            Path("chunk0.mp4"),
            0.0,
            2.0,
            0,
            0,
            12,
            processing_fps=6.0,
        ),
        Sam3VideoChunk(1, Path("chunk1.mp4"), 2.0, 2.0, 60, 60, 60),
    ]
    adapter = DummyAdapter()

    results = track_and_merge_sam3_video_chunks(
        adapter,  # type: ignore[arg-type]
        chunks,
        prompt="person",
        fps=30.0,
    )

    assert len(results) == 2
    assert [call["fps"] for call in adapter.calls] == [6.0, 30.0]
    assert [call["max_frames"] for call in adapter.calls] == [12, 60]


def test_merge_sam3_chunk_results_allocates_global_id_for_unmatched_collision() -> None:
    chunks = [
        Sam3VideoChunk(0, Path("chunk0.mp4"), 0.0, 4.0, 0, 0, 4),
        Sam3VideoChunk(1, Path("chunk1.mp4"), 2.0, 4.0, 2, 4, 4),
    ]
    chunk_results = [
        [
            frame_result(
                index,
                [
                    make_bbox_track("p0:0", (0, 0, 10, 10)),
                    make_bbox_track("p0:1", (20, 20, 30, 30)),
                ],
            )
            for index in range(4)
        ],
        [
            frame_result(
                index,
                [
                    make_bbox_track("p0:0", (100, 100, 110, 110)),
                    make_bbox_track("p0:1", (120, 120, 130, 130)),
                ],
            )
            for index in range(4)
        ],
    ]

    merged = merge_sam3_chunk_results(chunks, chunk_results, fps=1.0)

    assert [result.frame_index for result in merged] == [0, 1, 2, 3, 4, 5]
    assert [[track.track_id for track in result.tracks] for result in merged[:4]] == [
        ["p0:0", "p0:1"],
        ["p0:0", "p0:1"],
        ["p0:0", "p0:1"],
        ["p0:0", "p0:1"],
    ]
    assert [[track.track_id for track in result.tracks] for result in merged[4:]] == [
        ["p0:2", "p0:3"],
        ["p0:2", "p0:3"],
    ]


def test_infer_frame_sequence_fps_from_sampled_timestamps() -> None:
    frames = [
        SampledFrame(Path("0.jpg"), 10.0),
        SampledFrame(Path("1.jpg"), 10.5),
        SampledFrame(Path("2.jpg"), 11.0),
    ]

    assert infer_frame_sequence_fps(frames) == 2.0


def test_sam3_tracked_clip_frame_record_uses_roi_crop_when_roi_is_set() -> None:
    source_frame = SampledFrame(Path("source.jpg"), 5.0)
    crop_track = make_track("2", 5.0)
    full_frame_track = Track(
        track_id="2",
        bbox=BoundingBox(100, 200, 110, 220),
        label="person",
        confidence=0.9,
        timestamp_seconds=5.0,
    )
    frame_result = Sam3FrameResult(
        timestamp_seconds=5.0,
        image_path=Path("source_roi.jpg"),
        tracks=[crop_track],
    )

    record = sam3_tracked_clip_frame_record(
        source_frame,
        frame_result,
        full_frame_tracks=[full_frame_track],
        roi=(100, 200, 300, 400),
    )

    assert record.image_path == Path("source_roi.jpg")
    assert record.tracks == [crop_track]
