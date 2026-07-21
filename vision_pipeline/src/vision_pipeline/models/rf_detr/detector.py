"""RF-DETR detector adapter."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from vision_pipeline.core.schemas import BoundingBox, Detection
from vision_pipeline.models.rf_detr.schemas import RFDETRConfig


class RFDETRDetector:
    """Adapter boundary for RF-DETR detection."""

    _MODEL_CLASSES = {
        "nano": "RFDETRNano",
        "small": "RFDETRSmall",
        "medium": "RFDETRMedium",
        "large": "RFDETRLarge",
    }

    def __init__(self, config: RFDETRConfig) -> None:
        self.config = config
        self._model: Any | None = None

    def detect(self, frame: object, timestamp_seconds: float) -> list[Detection]:
        """Run RF-DETR on an RGB image array or image path."""
        return self.detect_image(frame, timestamp_seconds)

    def detect_image(self, image: object, timestamp_seconds: float = 0.0) -> list[Detection]:
        """Run detection on an image path, PIL image, RGB ndarray, or tensor."""
        model = self._load_model()
        predict_kwargs: dict[str, object] = {
            "threshold": self.config.confidence_threshold,
            "include_source_image": self.config.include_source_image,
        }
        if self.config.input_size > 0:
            predict_kwargs["shape"] = (self.config.input_size, self.config.input_size)

        predictions = model.predict(image, **predict_kwargs)
        return self._convert_predictions(predictions, timestamp_seconds)

    def detect_image_path(
        self,
        image_path: str | Path,
        timestamp_seconds: float = 0.0,
    ) -> list[Detection]:
        """Run detection on an image file path."""
        return self.detect_image(str(image_path), timestamp_seconds)

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            import rfdetr
        except ImportError as exc:
            raise RuntimeError(
                "RF-DETR inference requires the `rfdetr` package. "
                "Install it in a Python>=3.10 GPU environment with "
                "`pip install rfdetr supervision opencv-python`."
            ) from exc

        if self.config.weights_path:
            rf_detr_base = getattr(rfdetr, "RFDETR", None)
            if rf_detr_base is None or not hasattr(rf_detr_base, "from_checkpoint"):
                raise RuntimeError("Installed `rfdetr` does not expose RFDETR.from_checkpoint.")
            self._model = rf_detr_base.from_checkpoint(self.config.weights_path)
        else:
            class_name = self._MODEL_CLASSES.get(self.config.model_size.lower())
            if class_name is None:
                supported = ", ".join(sorted(self._MODEL_CLASSES))
                raise ValueError(
                    f"Unsupported RF-DETR model_size={self.config.model_size!r}; "
                    f"supported values: {supported}"
                )
            model_class = getattr(rfdetr, class_name)
            self._model = model_class()

        if self.config.optimize_for_inference:
            self._model.optimize_for_inference()

        return self._model

    def _convert_predictions(
        self,
        predictions: object,
        timestamp_seconds: float,
    ) -> list[Detection]:
        xyxy = _as_list(getattr(predictions, "xyxy", []))
        confidences = _as_list(getattr(predictions, "confidence", []))
        class_ids = _as_list(getattr(predictions, "class_id", []))
        class_names = self._prediction_class_names(predictions, class_ids)

        detections: list[Detection] = []
        for index, bbox_values in enumerate(xyxy):
            label = class_names[index] if index < len(class_names) else ""
            if not self._include_label(label):
                continue

            confidence = (
                float(confidences[index])
                if index < len(confidences) and confidences[index] is not None
                else 0.0
            )
            class_id = class_ids[index] if index < len(class_ids) else None
            x1, y1, x2, y2 = [float(value) for value in bbox_values]
            detections.append(
                Detection(
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    label=label,
                    confidence=confidence,
                    model_id=self.config.model_id,
                    metadata={
                        "class_id": _to_builtin(class_id),
                        "timestamp_seconds": timestamp_seconds,
                        "model_size": self.config.model_size,
                    },
                )
            )
        return detections

    def _prediction_class_names(
        self,
        predictions: object,
        class_ids: list,
    ) -> list[str]:
        data = getattr(predictions, "data", {}) or {}
        if "class_name" in data:
            return [str(value) for value in _as_list(data["class_name"])]

        model = self._model
        class_names = getattr(model, "class_names", None)
        if isinstance(class_names, dict):
            return [str(class_names.get(_to_builtin(class_id), "")) for class_id in class_ids]
        if isinstance(class_names, (list, tuple)):
            labels = []
            for class_id in class_ids:
                index = int(class_id)
                labels.append(str(class_names[index]) if 0 <= index < len(class_names) else "")
            return labels

        return [str(class_id) for class_id in class_ids]

    def _include_label(self, label: str) -> bool:
        normalized = label.strip().lower()
        allowed = {item.lower() for item in self.config.allowed_classes}
        denied = {item.lower() for item in self.config.denied_classes}
        if normalized in denied:
            return False
        if allowed and normalized not in allowed:
            return False
        return True


def _as_list(values: object) -> list:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        converted = values.tolist()
        return converted if isinstance(converted, list) else [converted]
    if isinstance(values, list):
        return values
    if isinstance(values, tuple):
        return list(values)
    if isinstance(values, Iterable) and not isinstance(values, (str, bytes, dict)):
        return list(values)
    return [values]


def _to_builtin(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()
    return value
