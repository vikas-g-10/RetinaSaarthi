
from ml.quality.service import assess_image


VALID_IMAGE = "datasets/aptos/train_images/000c1434d8d7.png"


def test_valid_image_without_thresholds_returns_manual_review():
    assessment = assess_image(VALID_IMAGE)

    assert assessment.metrics is not None
    assert assessment.result.decision == "MANUAL_REVIEW"
    assert "not been calibrated" in assessment.result.reason


def test_missing_image_returns_manual_review():
    assessment = assess_image("datasets/does_not_exist.png")

    assert assessment.metrics is None
    assert assessment.result.decision == "MANUAL_REVIEW"
    assert "could not be calculated" in assessment.result.reason


def test_explicit_technical_failure_returns_recapture():
    assessment = assess_image(
        VALID_IMAGE,
        confirmed_technical_failure=True,
    )

    assert assessment.metrics is not None
    assert assessment.result.decision == "RECAPTURE"


def test_uncertainty_returns_manual_review():
    assessment = assess_image(
        VALID_IMAGE,
        uncertainty_reasons=("Image visibility is uncertain.",),
    )

    assert assessment.metrics is not None
    assert assessment.result.decision == "MANUAL_REVIEW"
    assert "visibility is uncertain" in assessment.result.reason


def test_successful_assessment_returns_metrics():
    assessment = assess_image(VALID_IMAGE)

    assert assessment.metrics is not None
    assert assessment.metrics.width == 3216
    assert assessment.metrics.height == 2136
