
"""Conservative decision logic for the RetinaSaarthi quality gate.

Thresholds must be calibrated before they can support PASS decisions.
Field visibility must also be explicitly confirmed.

This is an engineering prototype, not a clinically validated system.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ml.quality.metrics import QualityMetrics


@dataclass(frozen=True)
class QualityThresholds:
    """Limits established through a documented calibration process."""

    min_sharpness: float
    min_illumination_mean: float
    max_illumination_mean: float
    min_contrast_p5_p95: float
    max_illumination_quadrant_ratio: float

    def __post_init__(self) -> None:
        values = (
            self.min_sharpness,
            self.min_illumination_mean,
            self.max_illumination_mean,
            self.min_contrast_p5_p95,
            self.max_illumination_quadrant_ratio,
        )

        if not all(isfinite(value) for value in values):
            raise ValueError("All quality thresholds must be finite.")

        if self.min_illumination_mean > self.max_illumination_mean:
            raise ValueError(
                "Minimum illumination cannot exceed maximum illumination."
            )


@dataclass(frozen=True)
class GateResult:
    """Outcome and explanation returned by the quality gate."""

    decision: str
    reason: str


def evaluate_quality(
    metrics: QualityMetrics | None,
    thresholds: QualityThresholds | None = None,
    *,
    confirmed_technical_failure: bool = False,
    uncertainty_reasons: tuple[str, ...] = (),
    field_assessment_confirmed: bool = False,
) -> GateResult:
    """Evaluate image quality conservatively.

    RECAPTURE requires an explicitly confirmed technical failure.
    PASS requires supplied calibrated thresholds and confirmed
    retinal field visibility.
    """

    if confirmed_technical_failure:
        return GateResult(
            decision="RECAPTURE",
            reason="A severe technical failure was explicitly confirmed.",
        )

    if uncertainty_reasons:
        return GateResult(
            decision="MANUAL_REVIEW",
            reason="; ".join(uncertainty_reasons),
        )

    if metrics is None:
        return GateResult(
            decision="MANUAL_REVIEW",
            reason="Image-quality metrics are unavailable.",
        )

    if thresholds is None:
        return GateResult(
            decision="MANUAL_REVIEW",
            reason="Quality thresholds have not been calibrated.",
        )

    measured_values = (
        metrics.sharpness,
        metrics.illumination_mean,
        metrics.illumination_std,
        metrics.illumination_quadrant_ratio,
        metrics.contrast_p5_p95,
        metrics.field_coverage_percent,
    )

    if not all(isfinite(value) for value in measured_values):
        return GateResult(
            decision="MANUAL_REVIEW",
            reason="One or more quality measurements are invalid or non-finite.",
        )

    if not field_assessment_confirmed:
        return GateResult(
            decision="MANUAL_REVIEW",
            reason="Retinal field visibility has not been confirmed.",
        )

    failures = []

    if metrics.sharpness < thresholds.min_sharpness:
        failures.append("Sharpness is below the calibrated limit.")

    if not (
        thresholds.min_illumination_mean
        <= metrics.illumination_mean
        <= thresholds.max_illumination_mean
    ):
        failures.append("Illumination is outside calibrated limits.")

    if metrics.contrast_p5_p95 < thresholds.min_contrast_p5_p95:
        failures.append("Contrast is below the calibrated limit.")

    if (
        metrics.illumination_quadrant_ratio
        > thresholds.max_illumination_quadrant_ratio
    ):
        failures.append(
            "Quadrant illumination variation exceeds the calibrated limit."
        )

    if failures:
        return GateResult(
            decision="MANUAL_REVIEW",
            reason=" ".join(failures),
        )

    return GateResult(
        decision="PASS",
        reason=(
            "Measurements are within supplied calibrated limits, "
            "and retinal field visibility has been confirmed."
        ),
    )
