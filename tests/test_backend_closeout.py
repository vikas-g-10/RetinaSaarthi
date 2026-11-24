"""Closeout tests for upload limits, model failures, RPC failures, and IDs."""

import asyncio
from io import BytesIO
from urllib.error import URLError

import pytest
from fastapi import HTTPException, UploadFile
from PIL import Image

from backend import main
from ml.classification.inference import reset_inference_service


def png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (32, 32), "black").save(buffer, format="PNG")
    return buffer.getvalue()


class TrackingReader(BytesIO):
    def __init__(self, value: bytes):
        super().__init__(value)
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        chunk = super().read(size)
        self.bytes_read += len(chunk)
        return chunk


def fake_assessment(decision: str = "PASS"):
    return type("Assessment", (), {
        "result": type("Result", (), {"decision": decision, "reason": "test"})(),
        "metrics": None,
    })()


def fake_prediction():
    return type("Prediction", (), {
        "predicted_class": 0,
        "predicted_label": "No DR",
        "probabilities": [0.2] * 5,
        "model_name": "test-model",
        "model_version": "test-version",
    })()


def test_chunked_upload_reader_aborts_near_limit():
    reader = TrackingReader(b"x" * (main.MAX_UPLOAD_SIZE + 1024 * 1024))
    upload = UploadFile(reader, filename="large.png")

    with pytest.raises(HTTPException) as error:
        asyncio.run(main._read_upload(upload))

    assert error.value.status_code == 413
    assert main.MAX_UPLOAD_SIZE < reader.bytes_read <= main.MAX_UPLOAD_SIZE + 1024 * 1024


def test_predict_returns_503_for_bad_checkpoint_and_does_not_write(monkeypatch):
    monkeypatch.setenv("RETINASAARTHI_CHECKPOINT", "/tmp/does-not-exist-retinasaarthi.pth")
    reset_inference_service()
    monkeypatch.setattr(main, "assess_image", lambda _path: fake_assessment())
    monkeypatch.setattr(main, "save_classification_result", lambda *_args: pytest.fail("must not write"))

    with pytest.raises(HTTPException) as error:
        asyncio.run(main.predict_classification(
            UploadFile(BytesIO(png_bytes()), filename="image.png"),
            None,
            {"id": "user", "token": "token"},
        ))

    assert error.value.status_code == 503
    reset_inference_service()


def test_failing_rpc_returns_502_without_success_payload(monkeypatch):
    monkeypatch.setattr(main, "assess_image", lambda _path: fake_assessment())
    monkeypatch.setattr(main, "get_inference_service", lambda: type(
        "Service", (), {"predict": lambda _self, _image: fake_prediction()}
    )())
    monkeypatch.setattr(main, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError("rpc down")))

    with pytest.raises(HTTPException) as error:
        asyncio.run(main.predict_classification(
            UploadFile(BytesIO(png_bytes()), filename="image.png"),
            None,
            {"id": "user", "token": "token"},
        ))

    assert error.value.status_code == 502
    assert "saved atomically" in error.value.detail


def test_success_returns_classification_result_id(monkeypatch):
    monkeypatch.setattr(main, "assess_image", lambda _path: fake_assessment())
    monkeypatch.setattr(main, "get_inference_service", lambda: type(
        "Service", (), {"predict": lambda _self, _image: fake_prediction()}
    )())
    monkeypatch.setattr(main, "save_classification_result", lambda *_args: "classification-result-id")

    response = asyncio.run(main.predict_classification(
        UploadFile(BytesIO(png_bytes()), filename="image.png"),
        None,
        {"id": "user", "token": "token"},
    ))

    assert response["result_id"] == "classification-result-id"
