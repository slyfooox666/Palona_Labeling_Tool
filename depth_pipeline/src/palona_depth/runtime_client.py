"""Safe localhost client for the shared AI_Runtime DA3 worker."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any
import urllib.error
import urllib.request

import numpy as np

from palona_depth.models import DepthArtifact


BASE_URL = "http://127.0.0.1:8765"
PINNED_DA3_REVISION = "f4a6c9b3c95e41c82048423d3493a81ec3fa810e"


class RuntimeClientError(RuntimeError):
    """Raised when the shared local runtime cannot complete a depth job."""


class AiModelsDepthClient:
    def __init__(self, *, device: str = "auto", timeout_seconds: float = 600.0) -> None:
        if device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError("device must be auto, cuda, mps, or cpu")
        self.device = device
        self.timeout_seconds = timeout_seconds
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.cli = resolve_ai_models_cli()

    def ensure_ready(self) -> dict[str, Any]:
        command = [str(self.cli), "use", "da3", "--device", self.device]
        completed = subprocess.run(command, text=True, capture_output=True, timeout=self.timeout_seconds)
        if completed.returncode != 0:
            raise RuntimeClientError(
                f"Could not start shared DA3 runtime: {completed.stderr.strip() or completed.stdout.strip()}"
            )
        try:
            health = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeClientError(f"Unexpected ai-models output: {completed.stdout!r}") from exc
        revision = health.get("model", {}).get("model_revision")
        if revision != PINNED_DA3_REVISION:
            raise RuntimeClientError(
                f"DA3 revision mismatch: expected {PINNED_DA3_REVISION}, received {revision}"
            )
        return health

    def infer_image(self, input_path: Path, output_dir: Path) -> DepthArtifact:
        output_dir = output_dir.expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        record = self._request(
            "POST",
            "/v1/jobs",
            {
                "task": "da3.depth_image",
                "input_path": str(input_path.expanduser().resolve()),
                "output_dir": str(output_dir),
                "options": {},
            },
        )
        job_id = str(record.get("job_id", ""))
        if not job_id:
            raise RuntimeClientError(f"Runtime did not return a job ID: {record}")
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            record = self._request("GET", f"/v1/jobs/{job_id}")
            status = record.get("status")
            if status == "succeeded":
                break
            if status == "failed":
                error = record.get("error") or {}
                raise RuntimeClientError(f"DA3 job {job_id} failed: {error.get('message', error)}")
            if time.monotonic() >= deadline:
                raise RuntimeClientError(f"Timed out waiting for DA3 job {job_id}")
            time.sleep(0.1)

        manifest_path = Path(record["result"]["manifest_path"]).resolve()
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            frame = manifest["frames"][0]
            depth_record = frame["depth"]
            depth_path = Path(depth_record["relative_depth_npy"]).resolve()
            confidence_value = depth_record.get("confidence_npy")
            confidence_path = Path(confidence_value).resolve() if confidence_value else None
            shape = tuple(int(value) for value in depth_record["shape"])
        except (OSError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeClientError(f"Invalid DA3 manifest {manifest_path}: {exc}") from exc
        if manifest.get("model", {}).get("model_revision") != PINNED_DA3_REVISION:
            raise RuntimeClientError(f"Unexpected DA3 model revision in {manifest_path}")
        return DepthArtifact(
            depth_path=depth_path,
            confidence_path=confidence_path,
            shape=(shape[0], shape[1]),
            model=dict(manifest.get("model") or {}),
            processing=dict(manifest.get("processing") or {}),
        )

    def stop(self) -> None:
        subprocess.run([str(self.cli), "stop"], text=True, capture_output=True, timeout=60, check=False)

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            BASE_URL + path,
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        try:
            with self.opener.open(request, timeout=30) as response:
                value = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeClientError(f"Runtime HTTP {exc.code} for {path}: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeClientError(f"Could not call shared runtime {path}: {exc}") from exc
        if not isinstance(value, dict):
            raise RuntimeClientError(f"Unexpected runtime response for {path}: {value!r}")
        return value


def resolve_ai_models_cli() -> Path:
    configured = os.environ.get("AI_MODELS_CLI")
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path(found) if (found := shutil.which("ai-models")) else None,
        Path.home() / ".local" / "bin" / "ai-models",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate.resolve()
    raise RuntimeClientError(
        "ai-models CLI was not found. Install the shared AI_Runtime or set AI_MODELS_CLI."
    )


def load_depth_array(artifact: DepthArtifact) -> tuple[np.ndarray, np.ndarray | None]:
    try:
        depth = np.load(artifact.depth_path, allow_pickle=False).astype(np.float32, copy=False)
        confidence = (
            np.load(artifact.confidence_path, allow_pickle=False).astype(np.float32, copy=False)
            if artifact.confidence_path is not None
            else None
        )
    except (OSError, ValueError) as exc:
        raise RuntimeClientError(f"Could not read DA3 artifacts: {exc}") from exc
    if depth.shape != artifact.shape or depth.ndim != 2 or not np.isfinite(depth).all():
        raise RuntimeClientError(
            f"Invalid DA3 depth array {artifact.depth_path}: shape={depth.shape}, expected={artifact.shape}"
        )
    if confidence is not None and confidence.shape != depth.shape:
        raise RuntimeClientError("DA3 confidence shape does not match depth shape")
    return depth, confidence
