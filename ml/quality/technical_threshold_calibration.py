
"""Technical calibration of quality metrics against synthetic degradations.

Exploratory engineering experiment only. NOT clinical validation.

All transformations of a source image remain in the same data split.
Existing datasets and reports are read-only.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from ml.quality.followup_phase2 import _fixed_metrics, cache_path
from ml.quality.run_quality_phase import image_files


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "quality_threshold_technical_v1"

SOURCE_CSV = REPORTS / "quality_metric_response_v2.csv"
LABEL_CSV = ROOT / "datasets/aptos/train.csv"

SAMPLE_SEED = 26038
SPLIT_SEED = 20261110
TEST_PER_GRADE = 12

DEGRADATIONS = ("reduced_contrast", "occlusion")


def select_source_images():
    labels = pd.read_csv(LABEL_CSV)
    grade_lookup = dict(
        zip(labels.id_code.astype(str), labels.diagnosis.astype(int))
    )

    paths = image_files(ROOT / "datasets/aptos/train_images")
    by_grade = {grade: [] for grade in range(5)}

    for path in paths:
        grade = grade_lookup.get(path.stem)
        if grade is not None:
            by_grade[grade].append(path)

    rng = random.Random(SAMPLE_SEED)
    selected = []

    for grade in range(5):
        selected.extend(
            rng.sample(by_grade[grade], min(40, len(by_grade[grade])))
        )

    if len(selected) != 200:
        raise RuntimeError(
            f"Expected 200 source images, found {len(selected)}"
        )

    return selected, grade_lookup


def create_source_split(selected, grade_lookup):
    rng = random.Random(SPLIT_SEED)
    rows = []

    for grade in range(5):
        group = [
            path for path in selected
            if grade_lookup[path.stem] == grade
        ]
        rng.shuffle(group)

        for index, path in enumerate(group):
            split = "heldout" if index < TEST_PER_GRADE else "calibration"
            rows.append({
                "file_name": path.name,
                "grade": grade,
                "split": split,
            })

    result = pd.DataFrame(rows)

    if result.file_name.nunique() != 200:
        raise RuntimeError("Source split contains duplicate filenames.")

    return result


def load_original_metrics(selected):
    rows = []

    for path in selected:
        cached = cache_path("APTOS", path.name)

        if not cached.exists():
            raise FileNotFoundError(cached)

        with np.load(cached) as data:
            metrics = _fixed_metrics(
                data["gray512"],
                data["green512"],
                data["mask512"].astype(bool),
            )

        row = {"file_name": path.name}
        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows)


def calibrate_threshold(original_values, degraded_values):
    """Choose metric direction and threshold using calibration images only."""

    original_values = np.asarray(original_values, dtype=float)
    degraded_values = np.asarray(degraded_values, dtype=float)

    y_true = np.concatenate([
        np.zeros(len(original_values), dtype=int),
        np.ones(len(degraded_values), dtype=int),
    ])

    scores = np.concatenate([original_values, degraded_values])

    if not np.isfinite(scores).all():
        raise ValueError("Non-finite metric values encountered.")

    auc_higher = roc_auc_score(y_true, scores)
    auc_lower = roc_auc_score(y_true, -scores)

    if auc_higher >= auc_lower:
        direction = "higher"
        oriented_scores = scores
        best_auc = auc_higher
    else:
        direction = "lower"
        oriented_scores = -scores
        best_auc = auc_lower

    # AUC of 0.5 means this metric does not discriminate the two groups.
    if best_auc <= 0.500001:
        return None

    fpr, tpr, thresholds = roc_curve(y_true, oriented_scores)
    best_index = int(np.argmax(tpr - fpr))

    oriented_threshold = float(thresholds[best_index])

    # Store threshold in the original metric scale.
    original_threshold = (
        oriented_threshold
        if direction == "higher"
        else -oriented_threshold
    )

    return {
        "direction": direction,
        "threshold": original_threshold,
        "calibration_auc": float(best_auc),
        "calibration_sensitivity": float(tpr[best_index]),
        "calibration_specificity": float(1 - fpr[best_index]),
    }


def evaluate_threshold(
    original_values,
    degraded_values,
    direction,
    threshold,
):
    original_values = np.asarray(original_values, dtype=float)
    degraded_values = np.asarray(degraded_values, dtype=float)

    if direction == "higher":
        original_scores = original_values
        degraded_scores = degraded_values
        oriented_threshold = threshold
    else:
        original_scores = -original_values
        degraded_scores = -degraded_values
        oriented_threshold = -threshold

    y_true = np.concatenate([
        np.zeros(len(original_scores), dtype=int),
        np.ones(len(degraded_scores), dtype=int),
    ])

    scores = np.concatenate([original_scores, degraded_scores])
    predictions = (scores >= oriented_threshold).astype(int)

    tn = int(((y_true == 0) & (predictions == 0)).sum())
    fp = int(((y_true == 0) & (predictions == 1)).sum())
    fn = int(((y_true == 1) & (predictions == 0)).sum())
    tp = int(((y_true == 1) & (predictions == 1)).sum())

    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")

    return {
        "original_images": len(original_values),
        "degraded_images": len(degraded_values),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    selected, grade_lookup = select_source_images()
    split = create_source_split(selected, grade_lookup)

    original = load_original_metrics(selected)
    response = pd.read_csv(SOURCE_CSV)

    expected_names = set(split.file_name)
    actual_names = set(response.file_name.unique())

    if expected_names != actual_names:
        raise RuntimeError(
            "Source image names in the existing experiment "
            "do not match the reconstructed 200-image sample."
        )

    degraded = response.pivot(
        index=["file_name", "degradation", "level"],
        columns="metric",
        values="value",
    ).reset_index()

    metrics = [
        column for column in original.columns
        if column != "file_name"
        and column in degraded.columns
        and pd.api.types.is_numeric_dtype(original[column])
    ]

    split.to_csv(OUTPUT / "source_image_split.csv", index=False)

    original.to_csv(
        OUTPUT / "original_fixed_mask_metrics.csv",
        index=False,
    )

    threshold_rows = []
    evaluation_rows = []

    calibration_names = set(
        split.loc[split.split == "calibration", "file_name"]
    )
    heldout_names = set(
        split.loc[split.split == "heldout", "file_name"]
    )

    for degradation in DEGRADATIONS:
        for metric in metrics:
            calibration_original = original[
                original.file_name.isin(calibration_names)
            ][metric].to_numpy()

            calibration_degraded = degraded[
                (degraded.file_name.isin(calibration_names))
                & (degraded.degradation == degradation)
            ][metric].to_numpy()

            if len(calibration_original) != 140:
                raise RuntimeError("Unexpected calibration source count.")

            if len(calibration_degraded) != 420:
                raise RuntimeError("Unexpected calibration degradation count.")

            calibrated = calibrate_threshold(
                calibration_original,
                calibration_degraded,
            )

            if calibrated is None:
                print(
                    f"SKIPPED: {degradation} / {metric} "
                    "(no calibration discrimination)"
                )
                continue

            threshold_rows.append({
                "degradation": degradation,
                "metric": metric,
                **calibrated,
            })

            heldout_original = original[
                original.file_name.isin(heldout_names)
            ][metric].to_numpy()

            for level in (1, 2, 3):
                heldout_degraded = degraded[
                    (degraded.file_name.isin(heldout_names))
                    & (degraded.degradation == degradation)
                    & (degraded.level == level)
                ][metric].to_numpy()

                if (
                    len(heldout_original) != 60
                    or len(heldout_degraded) != 60
                ):
                    raise RuntimeError("Unexpected held-out sample count.")

                result = evaluate_threshold(
                    heldout_original,
                    heldout_degraded,
                    calibrated["direction"],
                    calibrated["threshold"],
                )

                evaluation_rows.append({
                    "degradation": degradation,
                    "level": level,
                    "metric": metric,
                    "direction": calibrated["direction"],
                    "threshold": calibrated["threshold"],
                    **result,
                })

    thresholds = pd.DataFrame(threshold_rows)
    evaluation = pd.DataFrame(evaluation_rows)

    thresholds.to_csv(
        OUTPUT / "calibration_thresholds.csv",
        index=False,
    )

    evaluation.to_csv(
        OUTPUT / "heldout_technical_evaluation.csv",
        index=False,
    )

    print("\nSOURCE SPLIT")
    print(split.groupby(["split", "grade"]).size().to_string())

    print("\nCALIBRATION THRESHOLDS")
    if not thresholds.empty:
        print(thresholds.to_string(index=False))
    else:
        print("No metrics passed the calibration discrimination check.")

    print("\nHELD-OUT TECHNICAL EVALUATION")
    if not evaluation.empty:
        print(evaluation.to_string(index=False))
    else:
        print("No metrics available for held-out evaluation.")

    print("\nOUTPUT DIRECTORY:", OUTPUT)
    print("SOURCE IMAGES:", len(split))
    print("CALIBRATION IMAGES:", len(calibration_names))
    print("HELD-OUT IMAGES:", len(heldout_names))
    print("METRICS AVAILABLE:", len(metrics))
    print("CALIBRATED THRESHOLDS:", len(thresholds))
    print("\nNOTE: Synthetic technical evaluation only; not clinical validation.")


if __name__ == "__main__":
    main()
