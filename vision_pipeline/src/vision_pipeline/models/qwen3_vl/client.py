"""Qwen3-VL OpenAI-compatible HTTP client."""

from __future__ import annotations

import base64
import json
import mimetypes
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vision_pipeline.core.schemas import BoundingBox, Detection, VLMDecision
from vision_pipeline.core.tracklet import FrameEvidence
from vision_pipeline.models.qwen3_vl.schemas import Qwen3VLConfig


@dataclass(frozen=True)
class Qwen3VLImageResult:
    """Qwen3-VL result for one image/frame."""

    timestamp_seconds: float
    image_path: Path
    prompt: str
    raw_result: object
    raw_text: str
    detections: list[Detection]


class Qwen3VLClient:
    """Client for a self-hosted OpenAI-compatible Qwen3-VL endpoint."""

    def __init__(self, config: Qwen3VLConfig) -> None:
        self.config = config

    def detect_image_path(
        self,
        image_path: str | Path,
        target: str,
        timestamp_seconds: float = 0.0,
        bbox_coordinate_format: str | None = None,
    ) -> Qwen3VLImageResult:
        """Ask Qwen3-VL for prompt-grounded detections in one image."""
        path = Path(image_path)
        width, height = _image_size(path)
        prompt = _render_detection_prompt(
            self.config.detection_prompt_template,
            target=target,
            width=width,
            height=height,
        )
        result = self.analyze_image_path(
            path,
            prompt=prompt,
            timestamp_seconds=timestamp_seconds,
            response_format=self.config.response_format,
        )
        coordinate_format = bbox_coordinate_format or self.config.bbox_coordinate_format
        detections = self._detections_from_result(
            result.raw_result,
            target=target,
            timestamp_seconds=timestamp_seconds,
            image_size=(width, height),
            bbox_coordinate_format=coordinate_format,
        )
        return Qwen3VLImageResult(
            timestamp_seconds=timestamp_seconds,
            image_path=path,
            prompt=prompt,
            raw_result=result.raw_result,
            raw_text=result.raw_text,
            detections=detections,
        )

    def analyze_image_path(
        self,
        image_path: str | Path,
        prompt: str,
        timestamp_seconds: float = 0.0,
        response_format: str | None = None,
    ) -> Qwen3VLImageResult:
        """Run a single-image Qwen3-VL prompt and parse JSON when present."""
        path = Path(image_path)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(path)},
                    },
                ],
            }
        ]
        raw_response = self._post_chat_completion(
            messages=messages,
            response_format=response_format,
        )
        raw_text = _message_content(raw_response)
        parsed = _parse_json_content(raw_text)
        raw_result: object = parsed if parsed is not None else {"text": raw_text}
        return Qwen3VLImageResult(
            timestamp_seconds=timestamp_seconds,
            image_path=path,
            prompt=prompt,
            raw_result=raw_result,
            raw_text=raw_text,
            detections=[],
        )

    def classify_tracklet(
        self,
        track_id: str,
        prompt: str,
        evidence: list[FrameEvidence],
    ) -> VLMDecision:
        raw = self._post_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        *self._evidence_to_content(evidence),
                    ],
                }
            ],
            response_format=self.config.response_format,
        )
        content = _message_content(raw)
        result = _parse_json_content(content) or {"text": content}
        confidence = result.get("confidence") if isinstance(result, dict) else None
        return VLMDecision(
            track_id=track_id,
            prompt_id="runtime_prompt",
            model_id=self.config.model_id,
            result=result if isinstance(result, dict) else {"result": result},
            confidence=confidence,
            metadata={"raw_response": raw},
        )

    def _post_chat_completion(
        self,
        messages: list[dict],
        response_format: str | None,
    ) -> dict:
        payload: dict[str, Any] = {
            "model": self.config.model_name,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "messages": messages,
        }
        if response_format:
            payload["response_format"] = {"type": response_format}

        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        request = urllib.request.Request(
            url=self.config.endpoint.rstrip("/") + "/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request,
                timeout=self.config.timeout_seconds,
            ) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Qwen3-VL endpoint returned HTTP {exc.code}: {body}"
            ) from exc

    def _evidence_to_content(self, evidence: list[FrameEvidence]) -> list[dict]:
        content: list[dict] = []
        for frame in evidence:
            for crop in frame.crop_refs:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": crop.uri},
                    }
                )
        return content

    def _detections_from_result(
        self,
        result: object,
        target: str,
        timestamp_seconds: float,
        image_size: tuple[int, int],
        bbox_coordinate_format: str,
    ) -> list[Detection]:
        detections: list[Detection] = []
        for item in _iter_detection_items(result):
            bbox = _parse_bbox(
                _item_value(item, "bbox_xyxy", "bbox", "box", "rect"),
                image_size=image_size,
                coordinate_format=bbox_coordinate_format,
            )
            if bbox is None:
                continue
            label = str(_item_value(item, "label", "class", "name") or target)
            confidence = _parse_confidence(
                _item_value(item, "confidence", "score", "probability")
            )
            detections.append(
                Detection(
                    bbox=bbox,
                    label=label,
                    confidence=confidence,
                    model_id=self.config.model_id,
                    metadata={
                        "target": target,
                        "timestamp_seconds": timestamp_seconds,
                        "model_name": self.config.model_name,
                        "bbox_coordinate_format": bbox_coordinate_format,
                        "raw_detection": item,
                    },
                )
            )
        return detections


def _message_content(raw_response: dict) -> str:
    content = raw_response["choices"][0]["message"]["content"]
    if isinstance(content, str):
        return content
    return json.dumps(content)


def _render_detection_prompt(
    template: str,
    target: str,
    width: int,
    height: int,
) -> str:
    return (
        template.replace("{target}", target)
        .replace("{width}", str(width))
        .replace("{height}", str(height))
    )


def _image_data_url(image_path: Path) -> str:
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _image_size(image_path: Path) -> tuple[int, int]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Pillow is required to read image dimensions.") from exc

    with Image.open(image_path) as image:
        return image.size


def _parse_json_content(content: str) -> object | None:
    text = content.strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1).strip())
        except json.JSONDecodeError:
            pass

    candidate = _first_json_candidate(text)
    if candidate is not None:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    recovered_detections = _parse_partial_detection_content(text)
    if recovered_detections:
        return {
            "detections": recovered_detections,
            "parse_status": "partial_recovery",
            "parse_warning": (
                "Recovered complete detection objects from a response that was "
                "not valid JSON, often because the model output was truncated."
            ),
        }
    return None


def _first_json_candidate(text: str) -> str | None:
    start_indices = [index for index in (text.find("{"), text.find("[")) if index >= 0]
    if not start_indices:
        return None
    start = min(start_indices)
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _iter_detection_items(result: object) -> list[dict]:
    if isinstance(result, dict):
        for key in ("detections", "objects", "items"):
            value = result.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
        if any(key in result for key in ("bbox_xyxy", "bbox", "box", "rect")):
            return [result]
    if isinstance(result, list):
        return [item for item in result if isinstance(item, dict)]
    return []


def _parse_partial_detection_content(text: str) -> list[dict]:
    detections: list[dict] = []
    seen: set[tuple] = set()
    for match in re.finditer(r"\{", text):
        candidate = _balanced_json_object_from(text, match.start())
        if candidate is None:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        for item in _iter_detection_items(value):
            if not _looks_like_detection(item):
                continue
            key = _detection_identity(item)
            if key in seen:
                continue
            seen.add(key)
            detections.append(item)
    return detections


def _balanced_json_object_from(text: str, start: int) -> str | None:
    if start >= len(text) or text[start] != "{":
        return None

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _looks_like_detection(item: dict) -> bool:
    return (
        _item_value(item, "bbox_xyxy", "bbox", "box", "rect") is not None
        and _item_value(item, "label", "class", "name") is not None
    )


def _detection_identity(item: dict) -> tuple:
    label = _item_value(item, "label", "class", "name")
    bbox = _item_value(item, "bbox_xyxy", "bbox", "box", "rect")
    if isinstance(bbox, list):
        bbox_key = tuple(bbox)
    elif isinstance(bbox, tuple):
        bbox_key = bbox
    elif isinstance(bbox, dict):
        bbox_key = tuple(sorted(bbox.items()))
    else:
        bbox_key = bbox
    return (label, bbox_key)


def _item_value(item: dict, *keys: str) -> object | None:
    for key in keys:
        if key in item:
            return item[key]
    return None


def _parse_bbox(
    value: object,
    image_size: tuple[int, int],
    coordinate_format: str,
) -> BoundingBox | None:
    if value is None:
        return None
    if isinstance(value, dict):
        if all(key in value for key in ("x1", "y1", "x2", "y2")):
            raw_values = [value["x1"], value["y1"], value["x2"], value["y2"]]
        elif all(key in value for key in ("x", "y", "width", "height")):
            x = float(value["x"])
            y = float(value["y"])
            raw_values = [x, y, x + float(value["width"]), y + float(value["height"])]
        else:
            return None
    elif isinstance(value, (list, tuple)) and len(value) >= 4:
        raw_values = list(value[:4])
    else:
        return None

    x1, y1, x2, y2 = [float(raw_value) for raw_value in raw_values]
    width, height = image_size
    normalized = coordinate_format.strip().lower()
    if normalized in {"normalized", "normalized_0_1"}:
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    elif normalized in {"qwen1000", "normalized_0_1000", "0_1000"}:
        x1, x2 = x1 / 1000.0 * width, x2 / 1000.0 * width
        y1, y2 = y1 / 1000.0 * height, y2 / 1000.0 * height
    elif normalized == "auto":
        max_value = max(abs(x1), abs(y1), abs(x2), abs(y2))
        if max_value <= 1.5:
            x1, x2 = x1 * width, x2 * width
            y1, y2 = y1 * height, y2 * height
        elif max_value <= 1000 and (width > 1000 or height > 1000):
            x1, x2 = x1 / 1000.0 * width, x2 / 1000.0 * width
            y1, y2 = y1 / 1000.0 * height, y2 / 1000.0 * height

    x1 = min(max(0.0, x1), float(width))
    x2 = min(max(0.0, x2), float(width))
    y1 = min(max(0.0, y1), float(height))
    y2 = min(max(0.0, y2), float(height))
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    return BoundingBox(x1=left, y1=top, x2=right, y2=bottom)


def _parse_confidence(value: object) -> float:
    if value is None:
        return 0.0
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    if confidence > 1.0:
        return confidence / 100.0
    return confidence
