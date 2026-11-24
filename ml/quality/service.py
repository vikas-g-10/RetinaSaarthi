
"""Image assessment service for the RetinaSaarthi quality gate.

Connects image-quality metric calculation to the conservative
decision engine. This is an engineering prototype, not a
clinically validated service.
"""

from __future__ import annotations

from dataclasses import dataclass

from ml.quality.gate import (
    GateResult,
    QualityThresholds,
    evaluate_quality,
)
from ml.quality.metrics import QualityMetrics, compute_metrics


@dataclass(frozen=True)
class QualityAssessment:
    """Complete output of one image-quality assessment."""

    metrics: QualityMetrics | None
    result: GateResult


def assess_image(
    image_path: str,
    thresholds: QualityThresholds | None = None,
    *,
    confirmed_technical_failure: bool = False,
    uncertainty_reasons: tuple[str, ...] = (),
    field_assessment_confirmed: bool = False,
) -> QualityAssessment:
    """Calculate image metrics and evaluate the quality gate.

    A metric-calculation failure leads to MANUAL_REVIEW unless
    an explicit technical failure has already been confirmed.
    """

    try:
        metrics = compute_metrics(image_path)

    except (OSError, ValueError):
        result = evaluate_quality(
            None,
            thresholds,
            confirmed_technical_failure=confirmed_technical_failure,
            uncertainty_reasons=uncertainty_reasons,
            field_assessment_confirmed=field_assessment_confirmed,
        )

        if result.decision == "MANUAL_REVIEW":
            result = GateResult(
                decision="MANUAL_REVIEW",
                reason=(
                    "Image metrics could not be calculated. "
                    "Manual review is required."
                ),
            )

        return QualityAssessment(
            metrics=None,
            result=result,
        )

    result = evaluate_quality(
        metrics,
        thresholds,
        confirmed_technical_failure=confirmed_technical_failure,
        uncertainty_reasons=uncertainty_reasons,
        field_assessment_confirmed=field_assessment_confirmed,
    )

    return QualityAssessment(
        metrics=metrics,
        result=result,
    )
