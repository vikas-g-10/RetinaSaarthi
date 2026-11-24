
from math import nan

import pytest

from ml.quality.gate import (
    QualityThresholds,
    evaluate_quality,
)
from ml.quality.metrics import QualityMetrics


def make_metrics(**overrides):
    values = {
        "width": 1024,
        "height": 1024,
        "pixel_count": 1024 * 1024,
        "sharpness": 40.0,
        "illumination_mean": 100.0,
        "illumination_std": 20.0,
        "illumination_quadrant_ratio": 1.2,
        "contrast_p5_p95": 50.0,
        "field_coverage_percent": 85.0,
        "field_coverage_frame_percent": 85.0,
        "field_touches_edge": False,
        "touches_top": False,
        "touches_bottom": False,
        "touches_left": False,
        "touches_right": False,
    }

    values.update(overrides)
    return QualityMetrics(**values)


def make_thresholds():
    # Test-only values. These are NOT clinical thresholds.
    return QualityThresholds(
        min_sharpness=20.0,
        min_illumination_mean=50.0,
        max_illumination_mean=150.0,
        min_contrast_p5_p95=30.0,
        max_illumination_quadrant_ratio=2.0,
    )


def test_missing_metrics_requires_manual_review():
    result = evaluate_quality(None, make_thresholds())

    assert result.decision == "MANUAL_REVIEW"


def test_missing_thresholds_requires_manual_review():
    result = evaluate_quality(make_metrics())

    assert result.decision == "MANUAL_REVIEW"


def test_valid_metrics_pass_with_confirmed_field_visibility():
    result = evaluate_quality(
        make_metrics(),
        make_thresholds(),
        field_assessment_confirmed=True,
    )

    assert result.decision == "PASS"


def test_low_sharpness_requires_manual_review():
    metrics = make_metrics(sharpness=5.0)

    result = evaluate_quality(
        metrics,
        make_thresholds(),
        field_assessment_confirmed=True,
    )

    assert result.decision == "MANUAL_REVIEW"
    assert "Sharpness" in result.reason


def test_invalid_nan_metric_requires_manual_review():
    metrics = make_metrics(
        illumination_quadrant_ratio=nan
    )

    result = evaluate_quality(
        metrics,
        make_thresholds(),
        field_assessment_confirmed=True,
    )

    assert result.decision == "MANUAL_REVIEW"
    assert "invalid" in result.reason


def test_uncertainty_requires_manual_review():
    result = evaluate_quality(
        make_metrics(),
        make_thresholds(),
        uncertainty_reasons=("Retinal field detection is uncertain.",),
        field_assessment_confirmed=True,
    )

    assert result.decision == "MANUAL_REVIEW"


def test_confirmed_technical_failure_requires_recapture():
    result = evaluate_quality(
        None,
        confirmed_technical_failure=True,
    )

    assert result.decision == "RECAPTURE"


def test_edge_contact_alone_does_not_cause_recapture():
    metrics = make_metrics(
        field_touches_edge=True,
        touches_top=True,
    )

    result = evaluate_quality(
        metrics,
        make_thresholds(),
        field_assessment_confirmed=True,
    )

    assert result.decision == "PASS"


def test_unconfirmed_field_visibility_cannot_pass():
    result = evaluate_quality(
        make_metrics(),
        make_thresholds(),
        field_assessment_confirmed=False,
    )

    assert result.decision == "MANUAL_REVIEW"


def test_invalid_thresholds_are_rejected():
    with pytest.raises(ValueError):
        QualityThresholds(
            min_sharpness=20.0,
            min_illumination_mean=150.0,
            max_illumination_mean=50.0,
            min_contrast_p5_p95=30.0,
            max_illumination_quadrant_ratio=2.0,
        )


def test_non_finite_thresholds_are_rejected():
    with pytest.raises(ValueError):
        QualityThresholds(
            min_sharpness=nan,
            min_illumination_mean=50.0,
            max_illumination_mean=150.0,
            min_contrast_p5_p95=30.0,
            max_illumination_quadrant_ratio=2.0,
        )
