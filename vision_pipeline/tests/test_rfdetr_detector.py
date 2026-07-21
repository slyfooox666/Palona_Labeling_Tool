from __future__ import annotations

import sys
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_ROOT))

from vision_pipeline.models.rf_detr.detector import RFDETRDetector
from vision_pipeline.models.rf_detr.schemas import RFDETRConfig


class FakePredictions:
    xyxy = [[1, 2, 11, 22], [30, 40, 50, 60]]
    confidence = [0.91, 0.4]
    class_id = [0, 1]
    data = {"class_name": ["person", "chair"]}


def test_rfdetr_prediction_conversion_filters_allowed_classes() -> None:
    detector = RFDETRDetector(
        RFDETRConfig(
            model_id="rf_detr_test",
            model_size="large",
            allowed_classes=("person",),
        )
    )
    detections = detector._convert_predictions(FakePredictions(), timestamp_seconds=12.5)

    assert len(detections) == 1
    assert detections[0].label == "person"
    assert detections[0].bbox.as_xyxy() == (1.0, 2.0, 11.0, 22.0)
    assert detections[0].metadata["timestamp_seconds"] == 12.5
