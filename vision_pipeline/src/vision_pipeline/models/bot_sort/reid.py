"""ReID integration placeholder for BoT-SORT."""

from __future__ import annotations


class ReIDEncoder:
    """Adapter boundary for an appearance/ReID encoder."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def encode(self, crop: object) -> list[float]:
        raise NotImplementedError("Wire ReID model inference here.")
