
"""Run Phase 2 image-quality evidence and review steps.

Reads source datasets without modifying them. Reports and review images
are written under reports/. No clinical thresholds are selected.
"""

from __future__ import annotations

import argparse
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.quality.metrics import (  # noqa: E402
    compute_metrics,
    make_mask_overlay,
)

REPORTS = ROOT / "reports"
REVIEW = REPORTS / "quality_review"

METRIC_NAMES = [
    "sharpness",
    "illumination_mean",
    "illumination_std",
    "illumination_quadrant_ratio",
    "contrast_p5_p95",
    "field_coverage_percent",
]


def image_files(folder: Path) -> list[Path]:
    """Return all supported image files recursively."""
    if not folder.exists():
        return []

    return sorted(
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )


def inventories():
    """Return the image inventories and available DR grade mappings."""
    aptos = image_files(ROOT / "datasets/aptos/train_images")

    messidor = image_files(
        ROOT / "datasets/messidor2/extracted/IMAGES"
    )

    idrid = image_files(
        ROOT / "datasets/idrid/segmentation/extracted/"
        "A. Segmentation/1. Original Images"
    )

    aptos_labels = pd.read_csv(
        ROOT / "datasets/aptos/train.csv"
    )

    aptos_grade = dict(
        zip(
            aptos_labels.id_code.astype(str),
            aptos_labels.diagnosis.astype(int),
        )
    )

    messidor_labels = pd.read_csv(
        REPORTS / "messidor2_labels_verified.csv"
    )

    messidor_grade = dict(
        zip(
            messidor_labels.image_filename.astype(str).str.lower(),
            messidor_labels.dr_grade,
        )
    )

    return [
        ("APTOS", aptos, {"grade_map": aptos_grade}),
        ("Messidor-2", messidor, {"grade_map": messidor_grade}),
        ("IDRiD-segmentation", idrid, {"grade_map": {}}),
    ]


def metric_row(dataset: str, path: Path, meta: dict) -> dict:
    """Calculate one image's quality metrics and edge indicators."""
    result = compute_metrics(path)

    key = path.stem if dataset == "APTOS" else path.name.lower()
    grade = meta["grade_map"].get(key)

    return {
        "dataset": dataset,
        "file_name": str(path.relative_to(ROOT)),
        "grade": grade,
        "width": result.width,
        "height": result.height,
        "pixel_count": result.pixel_count,
        "sharpness": result.sharpness,
        "illumination_mean": result.illumination_mean,
        "illumination_std": result.illumination_std,
        "illumination_quadrant_ratio": result.illumination_quadrant_ratio,
        "contrast_p5_p95": result.contrast_p5_p95,
        "field_coverage_percent": result.field_coverage_percent,
        "field_touches_edge": result.field_touches_edge,
        "touches_top": result.touches_top,
        "touches_bottom": result.touches_bottom,
        "touches_left": result.touches_left,
        "touches_right": result.touches_right,
    }


def step1() -> None:
    """Calculate metrics, save overlays, failures, and correlations."""
    REVIEW.joinpath("masks").mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = []

    jobs = [
        (dataset, path, meta)
        for dataset, paths, meta in inventories()
        for path in paths
    ]

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(metric_row, dataset, path, meta): (dataset, path)
            for dataset, path, meta in jobs
        }

        for future in as_completed(futures):
            dataset, path = futures[future]

            try:
                rows.append(future.result())
            except Exception as exc:
                failures.append({
                    "dataset": dataset,
                    "file_name": str(path.relative_to(ROOT)),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })

    for dataset, paths, _ in inventories():
        for path in paths[:3]:
            try:
                output = (
                    REVIEW / "masks"
                    / f"{dataset.replace('-', '_')}_{path.stem}_mask.png"
                )

                cv2.imwrite(
                    str(output),
                    make_mask_overlay(path),
                )

            except Exception as exc:
                failures.append({
                    "dataset": dataset,
                    "file_name": str(path.relative_to(ROOT)),
                    "error_type": type(exc).__name__,
                    "error": f"overlay: {exc}",
                })

    output = REPORTS / "quality_metrics_standardised.csv"

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output, index=False)

    pd.DataFrame(
        failures,
        columns=["dataset", "file_name", "error_type", "error"],
    ).to_csv(
        REPORTS / "quality_metrics_failures.csv",
        index=False,
    )

    corr_rows = []

    aptos = metrics[
        metrics.dataset == "APTOS"
    ].dropna(subset=["grade"])

    messidor = metrics[
        metrics.dataset == "Messidor-2"
    ].dropna(subset=["grade"])

    for name in METRIC_NAMES:
        comparisons = [
            ("APTOS", "dr_grade", aptos, "grade"),
            ("APTOS", "pixel_count", aptos, "pixel_count"),
            ("Messidor-2", "pixel_count", messidor, "pixel_count"),
        ]

        for dataset, target, group, target_column in comparisons:
            valid = group[[name, target_column]].dropna()

            if len(valid) < 2:
                continue

            result = spearmanr(
                valid[name],
                valid[target_column],
            )

            corr_rows.append({
                "dataset": dataset,
                "target": target,
                "metric": name,
                "n": len(valid),
                "spearman_rho": result.statistic,
                "p_value": result.pvalue,
            })

    pd.DataFrame(corr_rows).to_csv(
        REPORTS / "quality_standardised_correlations.csv",
        index=False,
    )

    print(f"CSV={output}")
    print(f"ROWS_TOTAL={len(rows)}")

    if not metrics.empty:
        print(metrics.groupby("dataset").size().to_string())

    print(f"FAILURES={len(failures)}")
    print("DIRECTIONAL_EDGE_COLUMNS=touches_top,touches_bottom,touches_left,touches_right")
    print(pd.DataFrame(corr_rows).to_string(index=False))


def _degrade(image: np.ndarray, kind: str, level: int) -> np.ndarray:
    """Apply a controlled synthetic image degradation."""
    if kind == "blur":
        return cv2.GaussianBlur(
            image,
            (0, 0),
            sigmaX=(level + 1) * 1.5,
        )

    if kind == "darken":
        return np.clip(
            image.astype(np.float32) * (1.0 - 0.2 * level),
            0,
            255,
        ).astype(np.uint8)

    if kind == "reduced_contrast":
        return np.clip(
            (image.astype(np.float32) - 127.5)
            * (1.0 - 0.2 * level)
            + 127.5,
            0,
            255,
        ).astype(np.uint8)

    if kind == "occlusion":
        result = image.copy()
        height, width = result.shape[:2]
        fraction = 0.12 * level
        result[:max(1, int(height * fraction)), :] = 0
        return result

    raise ValueError(kind)


def step2() -> None:
    """Measure metric responses to controlled synthetic degradations."""
    rng = random.Random(26038)

    paths = image_files(ROOT / "datasets/aptos/train_images")
    labels = pd.read_csv(ROOT / "datasets/aptos/train.csv")

    grade_lookup = dict(
        zip(labels.id_code.astype(str), labels.diagnosis.astype(int))
    )

    by_grade = {grade: [] for grade in range(5)}

    for path in paths:
        grade = grade_lookup.get(path.stem)

        if grade is not None:
            by_grade[grade].append(path)

    sample = []

    for grade in range(5):
        sample.extend(
            rng.sample(
                by_grade[grade],
                min(40, len(by_grade[grade])),
            )
        )

    REVIEW.mkdir(parents=True, exist_ok=True)

    rows = []
    temp = REVIEW / "_temporary_degraded.png"

    try:
        for path in sample:
            original = cv2.imread(
                str(path),
                cv2.IMREAD_COLOR,
            )

            if original is None:
                continue

            grade = grade_lookup[path.stem]

            base = metric_row(
                "APTOS",
                path,
                {"grade_map": {path.stem: grade}},
            )

            for kind in (
                "blur",
                "darken",
                "reduced_contrast",
                "occlusion",
            ):
                for level in (1, 2, 3):
                    degraded = _degrade(original, kind, level)

                    cv2.imwrite(str(temp), degraded)
                    deg = compute_metrics(temp)

                    for metric in METRIC_NAMES:
                        original_value = base[metric]
                        degraded_value = getattr(deg, metric)

                        rows.append({
                            "seed": 26038,
                            "file_name": path.name,
                            "grade": grade,
                            "degradation": kind,
                            "level": level,
                            "metric": metric,
                            "original": original_value,
                            "degraded": degraded_value,
                            "change": degraded_value - original_value,
                        })

    finally:
        temp.unlink(missing_ok=True)

    response = pd.DataFrame(rows)

    response.to_csv(
        REPORTS / "quality_metric_response.csv",
        index=False,
    )

    summary_rows = []

    expected_direction = {
        "blur": {"sharpness": "decrease"},
        "darken": {"illumination_mean": "decrease"},
        "reduced_contrast": {"contrast_p5_p95": "decrease"},
        "occlusion": {"field_coverage_percent": "decrease"},
    }

    for (degradation, metric), group in response.groupby(
        ["degradation", "metric"]
    ):
        means = (
            group.groupby("level")["change"]
            .mean()
            .reindex([1, 2, 3])
            .to_numpy()
        )

        diffs = np.diff(means)

        monotonic = bool(
            np.all(diffs <= 0) or np.all(diffs >= 0)
        )

        summary_rows.append({
            "degradation": degradation,
            "metric": metric,
            "mean_change_level_1": means[0],
            "mean_change_level_2": means[1],
            "mean_change_level_3": means[2],
            "monotonic_mean_change": monotonic,
            "expected_direction_for_metric": (
                expected_direction.get(degradation, {}).get(
                    metric,
                    "not_primary",
                )
            ),
        })

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        REPORTS / "quality_metric_response_summary.csv",
        index=False,
    )

    print(f"SAMPLE_SIZE={len(sample)} SEED=26038")
    print(
        response.groupby(
            ["degradation", "level", "metric"]
        ).change.agg(["mean", "std"]).to_string()
    )
    print(summary.to_string(index=False))


def step3() -> None:
    """Generate visual montages and ungradable-image metric percentiles."""
    metrics = pd.read_csv(
        REPORTS / "quality_metrics_standardised.csv"
    )

    REVIEW.mkdir(parents=True, exist_ok=True)

    for metric in METRIC_NAMES:
        data = (
            metrics[
                metrics.dataset.isin(["APTOS", "Messidor-2"])
            ]
            .dropna(subset=[metric])
            .sort_values(metric)
            .reset_index(drop=True)
        )

        if data.empty:
            continue

        indices = sorted(set(
            list(range(min(5, len(data))))
            + list(range(max(0, len(data) - 5), len(data)))
            + [
                round(i * (len(data) - 1) / 20)
                for i in range(1, 20)
            ]
        ))

        chosen = data.iloc[indices].reset_index(drop=True)

        fig, axes = plt.subplots(
            5,
            6,
            figsize=(18, 16),
        )

        for ax, (_, row) in zip(axes.flat, chosen.iterrows()):
            image = cv2.imread(str(ROOT / row.file_name))

            if image is None:
                ax.axis("off")
                continue

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            ax.imshow(image)

            ax.set_title(
                f"{row.dataset}\n"
                f"{Path(row.file_name).name[:18]}\n"
                f"{metric}={row[metric]:.3f}",
                fontsize=7,
            )

            ax.axis("off")

        for ax in axes.flat[len(chosen):]:
            ax.axis("off")

        fig.suptitle(
            f"Quality review montage: {metric} (no threshold selected)"
        )

        fig.tight_layout()
        fig.savefig(
            REVIEW / f"montage_{metric}.png",
            dpi=150,
        )
        plt.close(fig)

    ungradable = [
        "20060411_58550_0200_PP.png",
        "IM002385.jpg",
        "IM003718.jpg",
        "IM004176.jpg",
    ]

    subset = metrics[
        metrics.file_name.str.lower().str.endswith(
            tuple(name.lower() for name in ungradable)
        )
    ].copy()

    reference = metrics[
        metrics.dataset.isin(["APTOS", "Messidor-2"])
    ]

    for metric in METRIC_NAMES:
        metric_values = reference[metric].dropna().to_numpy()

        subset[f"{metric}_percentile"] = subset[metric].apply(
            lambda value: (
                float(100 * np.mean(metric_values <= value))
                if len(metric_values) and pd.notna(value)
                else np.nan
            )
        )

    subset.to_csv(
        REPORTS / "ungradable_messidor2_quality_values.csv",
        index=False,
    )

    print(f"MONTAGES={len(METRIC_NAMES)}")

    columns = ["file_name"] + sum(
        ([metric, f"{metric}_percentile"] for metric in METRIC_NAMES),
        [],
    )

    print(subset[columns].to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "step",
        choices=("step1", "step2", "step3"),
    )

    args = parser.parse_args()

    {
        "step1": step1,
        "step2": step2,
        "step3": step3,
    }[args.step]()


if __name__ == "__main__":
    main()
