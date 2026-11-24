import asyncio
from io import BytesIO
from pathlib import Path

import pytest
from fastapi import HTTPException
from fastapi import UploadFile
from PIL import Image

from backend import main
from ml.classification.inference import EXPECTED_CLASSES, InferenceService, ModelUnavailable, reset_inference_service


CHECKPOINT = Path("outputs/classification/experiment3/best_model.pth")
FIXTURE = Path("datasets/aptos/train_images/000c1434d8d7.png")


@pytest.mark.skipif(not CHECKPOINT.exists(), reason="Experiment 3 checkpoint is absent; real inference test skipped.")
def test_real_checkpoint_shape_probabilities_mapping_and_determinism():
    service = InferenceService(CHECKPOINT)
    with Image.open(FIXTURE) as image:
        first = service.predict(image)
        second = service.predict(image)
    assert first.predicted_class in range(5)
    assert first.predicted_label == EXPECTED_CLASSES[first.predicted_class]
    assert len(first.probabilities) == 5
    assert sum(first.probabilities) == pytest.approx(1.0, abs=1e-6)
    assert first.probabilities == second.probabilities


def test_invalid_checkpoint_reports_model_unavailable(monkeypatch):
    monkeypatch.setenv("RETINASAARTHI_CHECKPOINT", "/tmp/retinasaarthi-missing-checkpoint.pth")
    reset_inference_service()
    status = main.model_status.__wrapped__() if hasattr(main.model_status, "__wrapped__") else asyncio.run(main.model_status())
    assert status["ready"] is False
    assert "unavailable" in status["error"].lower()
    reset_inference_service()


def test_missing_bearer_token_is_rejected():
    with pytest.raises(HTTPException) as error:
        main.require_researcher(None)
    assert error.value.status_code in (401, 503)


def test_recapture_short_circuits_before_inference(monkeypatch):
    image = Image.new("RGB", (32, 32), "black")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    upload = buffer.getvalue()
    monkeypatch.setattr(main, "assess_image", lambda _path: type("Assessment", (), {
        "result": type("Result", (), {"decision": "RECAPTURE", "reason": "Retake image"})(),
        "metrics": None,
    })())
    monkeypatch.setattr(main.get_inference_service(), "predict", lambda _image: pytest.fail("inference must not run"))
    response = asyncio.run(main.predict_classification(UploadFile(BytesIO(upload), filename="image.png"), None, {"id": "u", "token": "t"}))
    assert response["prediction"] is None
    assert response["quality"]["decision"] == "RECAPTURE"


def test_manual_review_short_circuits_before_inference(monkeypatch):
    image = Image.new("RGB", (32, 32), "black")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    upload = buffer.getvalue()
    monkeypatch.setattr(main, "assess_image", lambda _path: type("Assessment", (), {
        "result": type("Result", (), {"decision": "MANUAL_REVIEW", "reason": "Review image"})(),
        "metrics": None,
    })())
    monkeypatch.setattr(main.get_inference_service(), "predict", lambda _image: pytest.fail("inference must not run"))
    response = asyncio.run(main.predict_classification(UploadFile(BytesIO(upload), filename="image.png"), None, {"id": "u", "token": "t"}))
    assert response["prediction"] is None
    assert response["quality"]["decision"] == "MANUAL_REVIEW"


@pytest.mark.parametrize("patient_ref", ["", "\nname", "x" * 121])
def test_patient_reference_validation(patient_ref):
    image = Image.new("RGB", (32, 32), "black")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    upload = buffer.getvalue()
    with pytest.raises(HTTPException) as error:
        asyncio.run(main.predict_classification(UploadFile(BytesIO(upload), filename="image.png"), patient_ref, {"id": "u", "token": "t"}))
    assert error.value.status_code == 400
