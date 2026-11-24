"""CPU-only Phase 2 follow-up-2 audits.

This script preserves all earlier outputs, does not choose thresholds, and writes
only new reports/review files. It reuses the existing cache for reporting and
decodes only the fixed 200-image degradation sample and the selected coverage
extremes, with two CPU decoding workers.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.quality.metrics import WORKING_MAX_DIMENSION, find_field_mask  # noqa: E402
from ml.quality.run_quality_phase import _degrade, image_files, inventories  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
V2_REVIEW = REPORTS / "quality_review_v2"
CACHE = REPORTS / "quality_cache"


def sample_200() -> list[Path]:
    labels = pd.read_csv(ROOT / "datasets/aptos/train.csv")
    paths = image_files(ROOT / "datasets/aptos/train_images")
    by_grade = {g: [] for g in range(5)}
    for path in paths:
        by_grade[int(labels.loc[labels.id_code == path.stem, "diagnosis"].iloc[0])].append(path)
    rng = random.Random(26038)
    return sum((rng.sample(by_grade[g], 40) for g in range(5)), [])


def working_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"OpenCV could not decode image: {path}")
    height, width = image.shape[:2]
    scale = min(1.0, WORKING_MAX_DIMENSION / max(height, width))
    return (cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
            if scale < 1.0 else image)


def otsu_mask(working: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    _, candidate = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((candidate > 0).astype(np.uint8), 8)
    if count <= 1:
        raise ValueError("Otsu mask has no foreground component")
    return labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))


def frame_coverage(mask: np.ndarray) -> float:
    return 100.0 * float(mask.sum()) / mask.size


def load_cache_coverage(dataset: str, file_name: str) -> float:
    path = CACHE / dataset.replace("-", "_") / f"{Path(file_name).name}.npz"
    with np.load(path) as z:
        return float(z["field_coverage_frame_percent"])


def print_stratified() -> None:
    source = pd.read_csv(REPORTS / "quality_stratified_correlations.csv")
    keep = ["sharpness", "sharpness_256", "illumination_mean", "illumination_std", "contrast_p5_p95"]
    data = source[source.metric.isin(keep)].copy()
    wide = data.pivot_table(index=["dataset", "width", "height", "n", "grade_distribution"], columns="metric", values="spearman_rho", aggfunc="first").reset_index()
    groups = source[["dataset", "width", "height", "n", "grade_distribution"]].drop_duplicates()
    wide = groups.merge(wide, on=["dataset", "width", "height", "n", "grade_distribution"], how="left")
    for metric in keep:
        if metric not in wide:
            wide[metric] = np.nan
    wide = wide[["dataset", "width", "height", "n", "grade_distribution"] + keep]
    for _, row in wide.iterrows():
        if len(str(row.grade_distribution).split(";")) < 2:
            wide.loc[_, keep] = np.nan
    wide.to_csv(REPORTS / "quality_stratified_results_followup2.csv", index=False)
    print("STRATIFIED_RESULTS_FOLLOWUP2")
    print(wide.to_string(index=False))


def reestimated_degradations() -> None:
    rows = []
    for path in sample_200():
        original = cv2.imread(str(path), cv2.IMREAD_COLOR)
        working = working_image_from_array(original)
        original_mask = find_field_mask(working)
        original_cov = frame_coverage(original_mask)
        for kind in ("occlusion", "reduced_contrast"):
            for level in (1, 2, 3):
                degraded = _degrade(original, kind, level)
                degraded_working = working_image_from_array(degraded)
                degraded_mask = find_field_mask(degraded_working)
                rows.append({"seed": 26038, "file_name": path.name, "degradation": kind, "level": level,
                             "original_mask_coverage_percent": original_cov, "reestimated_mask_coverage_percent": frame_coverage(degraded_mask),
                             "coverage_change_percent_points": frame_coverage(degraded_mask) - original_cov,
                             "original_mask_area_pixels": int(original_mask.sum()), "reestimated_mask_area_pixels": int(degraded_mask.sum())})
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "quality_mask_reestimated_degradations_followup2.csv", index=False)
    summary = out.groupby(["degradation", "level"]).reestimated_mask_coverage_percent.agg(["mean", "std", "min", "max"]).reset_index()
    strict_rows = []
    for kind, group in out.groupby("degradation"):
        pivot = group.pivot(index="file_name", columns="level", values="reestimated_mask_coverage_percent")
        strict = int(((pivot[1] > pivot[2]) & (pivot[2] > pivot[3])).sum())
        strict_rows.append({"degradation": kind, "strictly_decreasing_images": strict, "images": len(pivot), "mean_level_1": pivot[1].mean(), "mean_level_2": pivot[2].mean(), "mean_level_3": pivot[3].mean(), "mean_strictly_decreasing": bool(pivot[1].mean() > pivot[2].mean() > pivot[3].mean())})
    strict_df = pd.DataFrame(strict_rows)
    strict_df.to_csv(REPORTS / "quality_mask_reestimated_degradation_summary_followup2.csv", index=False)
    print("REESTIMATED_DEGRADATION_SUMMARY")
    print(summary.to_string(index=False))
    print(strict_df.to_string(index=False))
    return out


def working_image_from_array(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    scale = min(1.0, WORKING_MAX_DIMENSION / max(height, width))
    return (cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))), interpolation=cv2.INTER_AREA)
            if scale < 1.0 else image)


def mask_robustness() -> None:
    rows = []
    for path in sample_200():
        original = cv2.imread(str(path), cv2.IMREAD_COLOR)
        working = working_image_from_array(original)
        current = find_field_mask(working)
        original_cov = frame_coverage(current)
        rows.append({"file_name": path.name, "degradation": "original", "level": 0, "original_mask_coverage_percent": original_cov, "reestimated_mask_coverage_percent": original_cov, "growth_percent_points": 0.0})
        for level in (1, 2, 3):
            degraded = working_image_from_array(_degrade(original, "reduced_contrast", level))
            degraded_mask = find_field_mask(degraded)
            coverage = frame_coverage(degraded_mask)
            rows.append({"file_name": path.name, "degradation": "reduced_contrast", "level": level, "original_mask_coverage_percent": original_cov, "reestimated_mask_coverage_percent": coverage, "growth_percent_points": coverage - original_cov})
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "quality_mask_reestimated_reduced_contrast_followup2.csv", index=False)
    summary = out[out.degradation == "reduced_contrast"].groupby("level").agg(mean_coverage_percent=("reestimated_mask_coverage_percent", "mean"), std_coverage_percent=("reestimated_mask_coverage_percent", "std"), mean_growth_percent_points=("growth_percent_points", "mean"), min_growth_percent_points=("growth_percent_points", "min"), max_growth_percent_points=("growth_percent_points", "max")).reset_index()
    summary.to_csv(REPORTS / "quality_mask_reestimated_reduced_contrast_summary_followup2.csv", index=False)
    print("REDUCED_CONTRAST_MASK_AUDIT")
    print(summary.to_string(index=False))
    print("MASK_GROWTH_AS_CONTRAST_FALLS", bool(summary.mean_growth_percent_points.is_monotonic_increasing and summary.mean_growth_percent_points.iloc[-1] > 0))


def mask_iou_audit() -> None:
    rows = []
    for path in sample_200():
        working = working_image(path)
        current = find_field_mask(working)
        alternate = otsu_mask(working)
        intersection = np.logical_and(current, alternate).sum()
        union = np.logical_or(current, alternate).sum()
        rows.append({"file_name": path.name, "current_mask_threshold": "max(5, percentile(gray, 1))", "otsu_threshold": "Otsu", "iou": float(intersection / union) if union else np.nan, "current_coverage_percent": frame_coverage(current), "otsu_coverage_percent": frame_coverage(alternate)})
    out = pd.DataFrame(rows).sort_values("iou")
    out.to_csv(REPORTS / "quality_mask_iou_otsu_followup2.csv", index=False)
    print("MASK_IOU_CURRENT_VS_OTSU")
    print(f"MEAN={out.iou.mean():.9f} MIN={out.iou.min():.9f} P5={out.iou.quantile(0.05):.9f}")
    print(out.head(10).to_string(index=False))


def coverage_extremes() -> None:
    rows = []
    standard = pd.read_csv(REPORTS / "quality_metrics_standardised.csv")
    for _, row in standard[standard.dataset.isin(["APTOS", "Messidor-2"])].iterrows():
        rows.append({"dataset": row.dataset, "file_name": row.file_name, "width": row.width, "height": row.height, "field_coverage_frame_percent": load_cache_coverage(row.dataset, row.file_name)})
    out = pd.DataFrame(rows)
    low = out.nsmallest(20, "field_coverage_frame_percent").assign(extreme="lowest20")
    high = out.nlargest(20, "field_coverage_frame_percent").assign(extreme="highest20")
    extremes = pd.concat([low, high], ignore_index=True)
    extremes.to_csv(REPORTS / "quality_field_coverage_extremes_followup2.csv", index=False)
    print("LOWEST_20_FRAME_COVERAGE")
    print(low.to_string(index=False))
    print("HIGHEST_20_FRAME_COVERAGE")
    print(high.to_string(index=False))
    target = V2_REVIEW / "masks"
    target.mkdir(parents=True, exist_ok=True)
    def overlay(row: dict) -> tuple[str, str]:
        path = ROOT / row["file_name"]
        working = working_image(path)
        mask = find_field_mask(working)
        boundary = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
        shown = working.copy()
        shown[boundary] = (0, 0, 255)
        output = target / f"{row['extreme']}_{row['dataset'].replace('-', '_')}_{Path(row['file_name']).stem}.png"
        cv2.imwrite(str(output), cv2.addWeighted(working, 0.75, shown, 0.25, 0))
        return row["file_name"], str(output)
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(overlay, extremes.to_dict("records")))
    print(f"OVERLAYS_SAVED={len(extremes)} DIRECTORY={target}")


def main() -> None:
    print_stratified()
    reestimated_degradations()
    mask_robustness()
    mask_iou_audit()
    coverage_extremes()


if __name__ == "__main__":
    main()
