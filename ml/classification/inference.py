"""Cached EfficientNet-B0 inference for the verified Experiment 3 checkpoint."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from ml.classification.dataset import CLASS_NAMES, get_transforms
from ml.classification.model import build_model


MODEL_NAME = "EfficientNet-B0"
EXPECTED_CLASSES = tuple(CLASS_NAMES)
DEFAULT_CHECKPOINT = "outputs/classification/experiment3/best_model.pth"


@dataclass(frozen=True)
class InferenceResult:
    predicted_class: int
    predicted_label: str
    probabilities: list[float]
    model_name: str
    model_version: str


class ModelUnavailable(RuntimeError):
    """Raised when the configured checkpoint cannot be loaded safely."""


class InferenceService:
    def __init__(self, checkpoint_path: str | Path | None = None) -> None:
        self.checkpoint_path = Path(
            checkpoint_path
            or os.getenv("RETINASAARTHI_CHECKPOINT", DEFAULT_CHECKPOINT)
        )
        if not self.checkpoint_path.is_absolute():
            self.checkpoint_path = Path(__file__).resolve().parents[2] / self.checkpoint_path
        self._model: torch.nn.Module | None = None
        self._classes: tuple[str, ...] | None = None
        self._version: str | None = None
        self._error: str | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self) -> None:
        if self._model is not None:
            return
        if self._error is not None:
            raise ModelUnavailable(self._error)
        try:
            if not self.checkpoint_path.is_file():
                raise ModelUnavailable("The configured classification model is unavailable.")
            checkpoint: Any = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )
            classes = tuple(checkpoint.get("class_names", ()))
            if classes != EXPECTED_CLASSES:
                raise ModelUnavailable("The classification model has an unsupported class mapping.")
            model = build_model(pretrained=False).to(self.device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            digest = hashlib.sha256(self.checkpoint_path.read_bytes()).hexdigest()[:12]
            epoch = checkpoint.get("epoch", "unknown")
            self._version = f"experiment3-epoch-{epoch}-{digest}"
            self._classes = classes
            self._model = model
        except ModelUnavailable as exc:
            self._error = str(exc)
            raise
        except Exception as exc:
            self._error = "The classification model could not be loaded."
            raise ModelUnavailable(self._error) from exc

    def status(self) -> dict[str, object]:
        try:
            self.load()
        except ModelUnavailable as exc:
            return {
                "ready": False,
                "model_name": MODEL_NAME,
                "model_version": None,
                "classes": list(EXPECTED_CLASSES),
                "error": str(exc),
            }
        return {
            "ready": True,
            "model_name": MODEL_NAME,
            "model_version": self._version,
            "classes": list(self._classes or EXPECTED_CLASSES),
        }

    def predict(self, image: Image.Image) -> InferenceResult:
        self.load()
        assert self._model is not None
        assert self._classes is not None
        tensor = get_transforms(training=False)(image.convert("RGB")).unsqueeze(0)
        tensor = tensor.to(self.device)
        with torch.no_grad():
            probabilities = torch.softmax(self._model(tensor), dim=1)[0].cpu()
        values = [float(value) for value in probabilities]
        index = int(probabilities.argmax().item())
        return InferenceResult(
            predicted_class=index,
            predicted_label=self._classes[index],
            probabilities=values,
            model_name=MODEL_NAME,
            model_version=self._version or "unknown",
        )


_service: InferenceService | None = None


def get_inference_service() -> InferenceService:
    global _service
    if _service is None:
        _service = InferenceService()
    return _service


def reset_inference_service() -> None:
    """Reset the cache for tests and explicit configuration changes."""
    global _service
    _service = None
