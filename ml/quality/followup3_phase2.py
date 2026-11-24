"""CPU-only Phase 2 follow-up-3 verification and mask audits."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import math
import random
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from ml.quality.metrics import WORKING_MAX_DIMENSION, find_field_mask  # noqa: E402
from ml.quality.run_quality_phase import _degrade, image_files  # noqa: E402
from ml.quality.followup2_phase2 import CACHE, V2_REVIEW, frame_coverage, sample_200  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
UNGRADABLE = REPORTS / "ungradable_messidor2_quality_values_v2.csv"


def working_from_array(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(1.0, WORKING_MAX_DIMENSION / max(h, w))
    return (cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
            if scale < 1.0 else image)


def otsu_mask(working: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)
    _, candidate = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, kernel)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats((candidate > 0).astype(np.uint8), 8)
    return labels == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))


def boundary_overlay(image: np.ndarray, mask: np.ndarray, colour: tuple[int, int, int]) -> np.ndarray:
    boundary = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((5, 5), np.uint8)) > 0
    out = image.copy()
    out[boundary] = colour
    return cv2.addWeighted(image, 0.75, out, 0.25, 0)


def print_ungradable_transposed() -> None:
    expected = {"20060411_58550_0200_PP.png", "IM002385.jpg", "IM003718.jpg", "IM004176.jpg"}
    frame = pd.read_csv(UNGRADABLE)
    print("UNGRADABLE_PANDAS_TRANSPOSED")
    for _, row in frame.iterrows():
        print(f"--- {row['file_name']} ---")
        print(row.to_string())
    actual = {Path(value).name for value in frame.file_name}
    case_insensitive = {value.lower() for value in actual} == {value.lower() for value in expected}
    check = pd.DataFrame([{"actual_file_name": value, "expected_label_name": next((x for x in expected if x.lower() == value.lower()), ""), "exact_case_match": value in expected, "case_insensitive_match": value.lower() in {x.lower() for x in expected}, "cause_if_not_exact": "Messidor-2 extension file uses uppercase .JPG; labels use lowercase .jpg" if value.lower().endswith(".jpg") and value not in expected else ""} for value in sorted(actual)])
    check.to_csv(REPORTS / "quality_ungradable_filename_check_followup3.csv", index=False)
    print("FILE_NAMES_EXACT_MATCH", actual == expected)
    print("FILE_NAMES_CASE_INSENSITIVE_MATCH", case_insensitive)
    print("ACTUAL_BASENAMES", sorted(actual))
    print("EXPECTED_BASENAMES", sorted(expected))


def omitted_aptos_groups() -> None:
    data = pd.read_csv(REPORTS / "quality_metrics_standardised.csv")
    aptos = data[data.dataset == "APTOS"]
    groups = []
    for (width, height), group in aptos.groupby(["width", "height"]):
        if len(group) < 30:
            dist = ";".join(f"{int(k)}:{int(v)}" for k, v in group.grade.value_counts().sort_index().items())
            groups.append({"width": width, "height": height, "n": len(group), "grade_distribution": dist})
    out = pd.DataFrame(groups).sort_values(["width", "height"])
    out.to_csv(REPORTS / "quality_aptos_omitted_size_groups_followup3.csv", index=False)
    print("APTOS_OMITTED_SIZE_GROUPS")
    print(out.to_string(index=False))
    print("OMITTED_IMAGES", int(out.n.sum()), "COVERED_IMAGES", int(aptos.shape[0] - out.n.sum()))


def load_coverage(dataset: str, file_name: str) -> float:
    path = CACHE / dataset.replace("-", "_") / f"{Path(file_name).name}.npz"
    with np.load(path) as z:
        return float(z["field_coverage_frame_percent"])


def coverage_by_group() -> None:
    standard = pd.read_csv(REPORTS / "quality_metrics_standardised.csv")
    rows = []
    for _, row in standard[standard.dataset.isin(["APTOS", "Messidor-2"])].iterrows():
        rows.append({"dataset": row.dataset, "width": row.width, "height": row.height, "coverage": load_coverage(row.dataset, row.file_name)})
    data = pd.DataFrame(rows)
    result = []
    for (dataset, width, height), group in data.groupby(["dataset", "width", "height"]):
        result.append({"dataset": dataset, "width": width, "height": height, "n": len(group), "min": group.coverage.min(), "median": group.coverage.median(), "max": group.coverage.max(), "coverage_ge_99_9": int((group.coverage >= 99.9).sum()), "coverage_eq_100": int(np.isclose(group.coverage, 100.0, rtol=0, atol=1e-9).sum())})
    out = pd.DataFrame(result).sort_values(["dataset", "width", "height"])
    out.to_csv(REPORTS / "quality_frame_coverage_by_group_followup3.csv", index=False)
    totals = data.groupby("dataset").coverage.agg(n="size", coverage_ge_99_9=lambda x: int((x >= 99.9).sum()), coverage_eq_100=lambda x: int(np.isclose(x, 100.0, rtol=0, atol=1e-9).sum())).reset_index()
    print("FRAME_COVERAGE_BY_GROUP")
    print(out.to_string(index=False))
    print("FRAME_COVERAGE_TOTALS")
    print(totals.to_string(index=False))
    print("TOTAL_EXACTLY_100", int(np.isclose(data.coverage, 100.0, rtol=0, atol=1e-9).sum()))


def save_iou_overlays() -> None:
    folder = V2_REVIEW / "masks_iou"
    folder.mkdir(parents=True, exist_ok=True)
    low = pd.read_csv(REPORTS / "quality_mask_iou_otsu_followup2.csv").sort_values("iou").head(10)
    def iou_overlay(name: str) -> str:
        path = ROOT / "datasets/aptos/train_images" / name
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        working = working_from_array(image)
        current = find_field_mask(working)
        otsu = otsu_mask(working)
        left = boundary_overlay(working, current, (0, 0, 255))
        right = boundary_overlay(working, otsu, (255, 0, 0))
        canvas = np.concatenate([left, right], axis=1)
        output = folder / f"iou_{Path(name).stem}.png"
        cv2.imwrite(str(output), canvas)
        return str(output)
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(iou_overlay, low.file_name.tolist()))

    growth = pd.read_csv(REPORTS / "quality_mask_reestimated_reduced_contrast_followup2.csv")
    largest = growth.loc[growth.growth_percent_points.idxmax(), "file_name"]
    chosen = []
    for path in sample_200():
        if path.name not in chosen:
            chosen.append(path.name)
        if len(chosen) == 5:
            break
    if largest not in chosen:
        chosen[-1] = largest
    def contrast_overlay(name: str) -> str:
        path = ROOT / "datasets/aptos/train_images" / name
        original = cv2.imread(str(path), cv2.IMREAD_COLOR)
        panels = []
        for level in (0, 1, 2, 3):
            image = original if level == 0 else _degrade(original, "reduced_contrast", level)
            working = working_from_array(image)
            mask = find_field_mask(working)
            panels.append(boundary_overlay(working, mask, (0, 0, 255)))
        canvas = np.concatenate(panels, axis=1)
        output = folder / f"reduced_contrast_masks_{Path(name).stem}.png"
        cv2.imwrite(str(output), canvas)
        return str(output)
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(contrast_overlay, chosen))
    print("IOU_OVERLAYS_SAVED", 10, "REDUCED_CONTRAST_OVERLAYS_SAVED", len(chosen), "LARGEST_GROWTH_IMAGE", largest)


def contrast_eroded() -> None:
    rows = []
    for path in sample_200():
        cache_path = CACHE / "APTOS" / f"{path.name}.npz"
        with np.load(cache_path) as z:
            original_gray = z["gray512"].astype(np.float32)
            original_mask = z["mask512"].astype(bool)
        original_contrast = float(np.percentile(original_gray[original_mask], 95) - np.percentile(original_gray[original_mask], 5))
        original = cv2.imread(str(path), cv2.IMREAD_COLOR)
        for level in (1, 2, 3):
            degraded = working_from_array(_degrade(original, "reduced_contrast", level))
            mask = find_field_mask(degraded)
            ys, xs = np.where(mask)
            crop = degraded[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            crop_mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1].astype(np.uint8)
            gray512 = cv2.cvtColor(cv2.resize(crop, (512, 512), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
            mask512 = cv2.resize(crop_mask, (512, 512), interpolation=cv2.INTER_NEAREST).astype(bool)
            radius = math.sqrt(mask512.sum() / math.pi)
            erosion_radius = max(1, int(math.ceil(0.03 * radius)))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erosion_radius + 1, 2 * erosion_radius + 1))
            eroded = cv2.erode(mask512.astype(np.uint8), kernel).astype(bool)
            contrast = float(np.percentile(gray512[eroded], 95) - np.percentile(gray512[eroded], 5))
            rows.append({"file_name": path.name, "level": level, "original_contrast": original_contrast, "reestimated_contrast": float(np.percentile(gray512[mask512], 95) - np.percentile(gray512[mask512], 5)), "eroded_contrast": contrast, "eroded_change_vs_original": contrast - original_contrast, "erosion_radius_pixels": erosion_radius})
    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / "quality_contrast_eroded_mask_followup3.csv", index=False)
    summary = out.groupby("level").agg(original_contrast=("original_contrast", "mean"), reestimated_contrast=("reestimated_contrast", "mean"), eroded_contrast=("eroded_contrast", "mean"), eroded_mean_change_vs_original=("eroded_change_vs_original", "mean"), eroded_std_change_vs_original=("eroded_change_vs_original", "std")).reset_index()
    summary.to_csv(REPORTS / "quality_contrast_eroded_mask_summary_followup3.csv", index=False)
    print("ERODED_CONTRAST_SUMMARY")
    print(summary.to_string(index=False))


def main() -> None:
    print_ungradable_transposed()
    omitted_aptos_groups()
    coverage_by_group()
    save_iou_overlays()
    contrast_eroded()


if __name__ == "__main__":
    main()
