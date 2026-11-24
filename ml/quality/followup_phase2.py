
"""Evidence-only follow-up analyses for the Phase 2 image-quality gate.

This module does not modify source datasets or select clinical thresholds.
It builds a regenerable per-image cache and performs additional analyses.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.quality.metrics import (  # noqa: E402
    STANDARD_SIZE,
    WORKING_MAX_DIMENSION,
    find_field_mask,
)

from ml.quality.run_quality_phase import (  # noqa: E402
    _degrade,
    image_files,
    inventories,
)

REPORTS = ROOT / "reports"
CACHE = REPORTS / "quality_cache"
V2_REVIEW = REPORTS / "quality_review_v2"

METRIC_NAMES = [
    "sharpness",
    "illumination_mean",
    "illumination_std",
    "illumination_quadrant_ratio",
    "contrast_p5_p95",
    "field_coverage_percent",
]

NEW_METRIC_NAMES = METRIC_NAMES + [
    "sharpness_256",
    "field_coverage_frame_percent",
    "bbox_aspect_ratio",
    "circularity",
]

EDGE_FIELDS = [
    "touches_top",
    "touches_bottom",
    "touches_left",
    "touches_right",
]


def _working_image(path: Path):
    """Prepare image arrays, mask, framing metrics, and edge indicators."""
    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            f"OpenCV could not decode image: {path}"
        )

    original_height, original_width = image.shape[:2]

    scale = min(
        1.0,
        WORKING_MAX_DIMENSION / max(original_height, original_width),
    )

    if scale < 1.0:
        working = cv2.resize(
            image,
            (
                max(1, round(original_width * scale)),
                max(1, round(original_height * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )
    else:
        working = image

    mask = find_field_mask(working)

    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        raise ValueError(f"Empty field mask: {path}")

    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1

    working_height, working_width = working.shape[:2]

    touches_top = bool(y0 == 0)
    touches_bottom = bool(y1 == working_height)
    touches_left = bool(x0 == 0)
    touches_right = bool(x1 == working_width)

    touches_edge = bool(
        touches_top
        or touches_bottom
        or touches_left
        or touches_right
    )

    crop = working[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1].astype(np.uint8)

    crop512 = cv2.resize(
        crop,
        (STANDARD_SIZE, STANDARD_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    mask512 = cv2.resize(
        crop_mask,
        (STANDARD_SIZE, STANDARD_SIZE),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    gray512 = cv2.cvtColor(
        crop512,
        cv2.COLOR_BGR2GRAY,
    )

    green512 = crop512[:, :, 1]

    gray256 = cv2.resize(
        gray512,
        (256, 256),
        interpolation=cv2.INTER_AREA,
    )

    green256 = cv2.resize(
        green512,
        (256, 256),
        interpolation=cv2.INTER_AREA,
    )

    mask256 = cv2.resize(
        mask512.astype(np.uint8),
        (256, 256),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    area = float(mask.sum())
    frame_area = float(mask.shape[0] * mask.shape[1])
    bbox_area = float((x1 - x0) * (y1 - y0))

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    perimeter = (
        float(
            cv2.arcLength(
                max(contours, key=cv2.contourArea),
                True,
            )
        )
        if contours
        else float("nan")
    )

    circularity = (
        float(4.0 * np.pi * area / (perimeter * perimeter))
        if perimeter > 0
        else float("nan")
    )

    frame_coverage = 100.0 * area / frame_area
    bbox_coverage = 100.0 * area / bbox_area
    aspect_ratio = (x1 - x0) / max(1, y1 - y0)

    return {
        "gray512": gray512,
        "green512": green512,
        "mask512": mask512,
        "gray256": gray256,
        "green256": green256,
        "mask256": mask256,
        "width": original_width,
        "height": original_height,
        "field_touches_edge": touches_edge,
        "touches_top": touches_top,
        "touches_bottom": touches_bottom,
        "touches_left": touches_left,
        "touches_right": touches_right,
        "field_coverage_frame_percent": frame_coverage,
        "bbox_aspect_ratio": aspect_ratio,
        "circularity": circularity,
        "field_coverage_percent": bbox_coverage,
    }


def _cache_one(dataset: str, path: Path) -> tuple[str, str]:
    """Create or update one image cache entry.

    Existing cache entries are preserved if they already contain the
    required directional edge fields. Older entries are regenerated.
    """
    target = (
        CACHE / dataset.replace("-", "_")
        / f"{path.name}.npz"
    )

    required_fields = {
        "touches_top",
        "touches_bottom",
        "touches_left",
        "touches_right",
    }

    if target.exists():
        try:
            with np.load(target) as existing:
                if required_fields.issubset(existing.files):
                    return dataset, str(target)
        except Exception:
            pass

    target.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    values = _working_image(path)

    np.savez_compressed(
        target,
        **values,
    )

    return dataset, str(target)


def build_cache() -> None:
    """Build or update APTOS and Messidor-2 image caches."""
    start = time.perf_counter()

    jobs = [
        (dataset, path)
        for dataset, paths, _ in inventories()
        if dataset in ("APTOS", "Messidor-2")
        for path in paths
    ]

    done = 0
    failures = []

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(_cache_one, dataset, path): (dataset, path)
            for dataset, path in jobs
        }

        for future in as_completed(futures):
            dataset, path = futures[future]

            try:
                future.result()
                done += 1

            except Exception as exc:
                failures.append({
                    "dataset": dataset,
                    "file_name": str(path.relative_to(ROOT)),
                    "error": str(exc),
                })

    pd.DataFrame(
        failures,
        columns=["dataset", "file_name", "error"],
    ).to_csv(
        REPORTS / "quality_cache_failures.csv",
        index=False,
    )

    size_bytes = (
        sum(p.stat().st_size for p in CACHE.rglob("*.npz"))
        if CACHE.exists()
        else 0
    )

    print(
        f"CACHE_IMAGES={done} "
        f"CACHE_FAILURES={len(failures)} "
        f"CACHE_BYTES={size_bytes} "
        f"CACHE_GIB={size_bytes / 1024**3:.3f} "
        f"BUILD_SECONDS={time.perf_counter() - start:.3f}"
    )


def cache_path(dataset: str, file_name: str) -> Path:
    return (
        CACHE / dataset.replace("-", "_")
        / f"{Path(file_name).name}.npz"
    )


def _cache_metric(path: Path) -> dict:
    """Calculate metrics from one cache entry."""
    with np.load(path) as z:
        gray = z["gray512"].astype(np.float32)
        green = z["green512"].astype(np.float32)
        mask = z["mask512"].astype(bool)

        values = gray[mask]

        if values.size == 0:
            raise ValueError(f"Empty cached mask: {path}")

        lap = cv2.Laplacian(
            green,
            cv2.CV_32F,
        )

        qmeans = []

        quadrants = (
            (0, 256, 0, 256),
            (0, 256, 256, 512),
            (256, 512, 0, 256),
            (256, 512, 256, 512),
        )

        for y0, y1, x0, x1 in quadrants:
            qmask = mask[y0:y1, x0:x1]

            if qmask.any():
                qvalues = gray[y0:y1, x0:x1][qmask]
                qmeans.append(float(qvalues.mean()))

        if len(qmeans) >= 2:
            quadrant_ratio = max(qmeans) / max(min(qmeans), 1e-6)
        else:
            quadrant_ratio = float("nan")

        out = {
            "sharpness": float(np.var(lap[mask])),
            "illumination_mean": float(values.mean()),
            "illumination_std": float(values.std()),
            "illumination_quadrant_ratio": quadrant_ratio,
            "contrast_p5_p95": float(
                np.percentile(values, 95)
                - np.percentile(values, 5)
            ),
            "field_coverage_percent": float(
                z["field_coverage_percent"]
            ),
            "field_coverage_frame_percent": float(
                z["field_coverage_frame_percent"]
            ),
            "bbox_aspect_ratio": float(
                z["bbox_aspect_ratio"]
            ),
            "circularity": float(z["circularity"]),
            "field_touches_edge": bool(
                z["field_touches_edge"]
            ),
            "touches_top": bool(z["touches_top"]),
            "touches_bottom": bool(z["touches_bottom"]),
            "touches_left": bool(z["touches_left"]),
            "touches_right": bool(z["touches_right"]),
        }

        small_lap = cv2.Laplacian(
            z["green256"].astype(np.float32),
            cv2.CV_32F,
        )

        out["sharpness_256"] = float(
            np.var(
                small_lap[z["mask256"].astype(bool)]
            )
        )

        return out


def _load_cache_metrics(standard: pd.DataFrame) -> pd.DataFrame:
    """Load cached measurements for all relevant standardised rows."""
    rows = []

    for _, row in standard[
        standard.dataset.isin(["APTOS", "Messidor-2"])
    ].iterrows():

        cache = _cache_metric(
            cache_path(row.dataset, row.file_name)
        )

        rows.append({
            "dataset": row.dataset,
            "file_name": row.file_name,
            **cache,
        })

    return pd.DataFrame(rows)


def full_correlations() -> None:
    """Calculate correlations using standardised and cached measurements."""
    standard = pd.read_csv(
        REPORTS / "quality_metrics_standardised.csv"
    )

    cache_df = _load_cache_metrics(standard)

    merged = standard.merge(
        cache_df[
            ["dataset", "file_name", "sharpness_256"]
        ],
        on=["dataset", "file_name"],
        how="left",
    )

    corr = []

    for metric in METRIC_NAMES + ["sharpness_256"]:
        aptos = merged[
            merged.dataset == "APTOS"
        ].dropna(subset=[metric, "grade"])

        for target in ("grade", "pixel_count"):
            valid = aptos[[metric, target]].dropna()

            if len(valid) < 2:
                continue

            result = spearmanr(
                valid[metric],
                valid[target],
            )

            corr.append({
                "dataset": "APTOS",
                "target": (
                    "dr_grade" if target == "grade"
                    else "pixel_count"
                ),
                "metric": metric,
                "n": len(valid),
                "spearman_rho": result.statistic,
                "p_value": result.pvalue,
            })

        messidor = merged[
            (merged.dataset == "Messidor-2")
            & merged.grade.notna()
        ].dropna(subset=[metric, "pixel_count"])

        if len(messidor) >= 2:
            result = spearmanr(
                messidor[metric],
                messidor.pixel_count,
            )

            corr.append({
                "dataset": "Messidor-2",
                "target": "pixel_count",
                "metric": metric,
                "n": len(messidor),
                "spearman_rho": result.statistic,
                "p_value": result.pvalue,
            })

    corr_df = pd.DataFrame(corr)

    corr_df.to_csv(
        REPORTS / "quality_standardised_correlations_v2.csv",
        index=False,
    )

    raw = pd.read_csv(
        REPORTS / "aptos_quality_metrics.csv"
    )

    raw_rows = []

    for target in ("diagnosis", "width_height_pixel_count"):
        x = (
            raw.diagnosis
            if target == "diagnosis"
            else raw.width * raw.height
        )

        result = spearmanr(raw.sharpness, x)

        raw_rows.append({
            "dataset": "APTOS",
            "target": (
                "dr_grade" if target == "diagnosis"
                else "pixel_count"
            ),
            "metric": "raw_sharpness",
            "n": len(raw),
            "spearman_rho": result.statistic,
            "p_value": result.pvalue,
        })

    raw_df = pd.DataFrame(raw_rows)

    raw_df.to_csv(
        REPORTS / "quality_raw_sharpness_correlations.csv",
        index=False,
    )

    print("FULL_STANDARDISED_CORRELATIONS")
    print(corr_df.to_string(index=False))
    print("RAW_SHARPNESS_CORRELATIONS")
    print(raw_df.to_string(index=False))


def stratified() -> None:
    """Check metric-grade associations within image-resolution groups."""
    standard = pd.read_csv(
        REPORTS / "quality_metrics_standardised.csv"
    )

    cache_df = _load_cache_metrics(standard)

    data = standard.merge(
        cache_df[
            ["dataset", "file_name", "sharpness_256"]
        ],
        on=["dataset", "file_name"],
        how="inner",
    )

    rows = []

    for dataset, group in data[
        data.grade.notna()
    ].groupby("dataset"):

        for (width, height), subset in group.groupby(
            ["width", "height"]
        ):

            if dataset == "APTOS" and len(subset) < 30:
                continue

            if (
                dataset == "Messidor-2"
                and (width, height) not in {
                    (1440, 960),
                    (2240, 1488),
                    (2304, 1536),
                }
            ):
                continue

            distribution = ";".join(
                f"{int(k)}:{int(v)}"
                for k, v in (
                    subset.grade.value_counts()
                    .sort_index()
                    .items()
                )
            )

            for metric in METRIC_NAMES + ["sharpness_256"]:
                valid = subset[[metric, "grade"]].dropna()

                if len(valid) < 2:
                    continue

                result = spearmanr(
                    valid[metric],
                    valid.grade,
                )

                rows.append({
                    "dataset": dataset,
                    "width": width,
                    "height": height,
                    "n": len(valid),
                    "grade_distribution": distribution,
                    "metric": metric,
                    "spearman_rho": result.statistic,
                    "p_value": result.pvalue,
                })

    output = pd.DataFrame(rows)

    output.to_csv(
        REPORTS / "quality_stratified_correlations.csv",
        index=False,
    )

    print("STRATIFIED_CORRELATIONS")
    print(output.to_string(index=False))


def _fixed_metrics(
    gray: np.ndarray,
    green: np.ndarray,
    mask: np.ndarray,
) -> dict:
    """Calculate metrics using an intentionally fixed mask."""
    values = gray[mask]

    if values.size == 0:
        raise ValueError("Empty fixed mask.")

    lap = cv2.Laplacian(
        green.astype(np.float32),
        cv2.CV_32F,
    )

    ys, xs = np.where(mask)

    x0 = int(xs.min())
    x1 = int(xs.max()) + 1
    y0 = int(ys.min())
    y1 = int(ys.max()) + 1

    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )

    perimeter = cv2.arcLength(
        max(contours, key=cv2.contourArea),
        True,
    )

    qmeans = []

    for ya, yb, xa, xb in (
        (0, 256, 0, 256),
        (0, 256, 256, 512),
        (256, 512, 0, 256),
        (256, 512, 256, 512),
    ):
        qmask = mask[ya:yb, xa:xb]

        if qmask.any():
            qmeans.append(
                float(gray[ya:yb, xa:xb][qmask].mean())
            )

    quadrant_ratio = (
        max(qmeans) / max(min(qmeans), 1e-6)
        if len(qmeans) >= 2
        else float("nan")
    )

    touches_top = bool(y0 == 0)
    touches_bottom = bool(y1 == 512)
    touches_left = bool(x0 == 0)
    touches_right = bool(x1 == 512)

    return {
        "sharpness": float(np.var(lap[mask])),
        "illumination_mean": float(values.mean()),
        "illumination_std": float(values.std()),
        "illumination_quadrant_ratio": quadrant_ratio,
        "contrast_p5_p95": float(
            np.percentile(values, 95)
            - np.percentile(values, 5)
        ),
        "field_coverage_frame_percent": (
            100.0 * float(mask.sum()) / mask.size
        ),
        "bbox_aspect_ratio": float(
            (x1 - x0) / max(1, y1 - y0)
        ),
        "circularity": (
            float(4 * np.pi * mask.sum() / (perimeter * perimeter))
            if perimeter > 0
            else float("nan")
        ),
        "field_touches_edge": bool(
            touches_top
            or touches_bottom
            or touches_left
            or touches_right
        ),
        "touches_top": touches_top,
        "touches_bottom": touches_bottom,
        "touches_left": touches_left,
        "touches_right": touches_right,
    }


def fixed_mask_diagnostics() -> None:
    """Evaluate synthetic degradation with the original mask held fixed."""
    labels = pd.read_csv(
        ROOT / "datasets/aptos/train.csv"
    )

    paths = image_files(
        ROOT / "datasets/aptos/train_images"
    )

    grade_lookup = dict(
        zip(labels.id_code.astype(str), labels.diagnosis.astype(int))
    )

    by_grade = {grade: [] for grade in range(5)}

    for path in paths:
        grade = grade_lookup.get(path.stem)

        if grade is not None:
            by_grade[grade].append(path)

    rng = random.Random(26038)

    sample = sum(
        (
            rng.sample(by_grade[grade], 40)
            for grade in range(5)
        ),
        [],
    )

    rows = []

    for path in sample:
        with np.load(cache_path("APTOS", path.name)) as z:
            gray = z["gray512"]
            green = z["green512"]
            mask = z["mask512"].astype(bool)

        for kind in ("reduced_contrast", "occlusion"):
            for level in (1, 2, 3):
                degraded_gray = _degrade(gray, kind, level)
                degraded_green = _degrade(green, kind, level)

                metrics = _fixed_metrics(
                    degraded_gray,
                    degraded_green,
                    mask,
                )

                for metric, value in metrics.items():
                    if isinstance(value, (bool, np.bool_)):
                        continue

                    rows.append({
                        "seed": 26038,
                        "file_name": path.name,
                        "degradation": kind,
                        "level": level,
                        "metric": metric,
                        "value": value,
                    })

    response = pd.DataFrame(rows)

    response.to_csv(
        REPORTS / "quality_metric_response_v2.csv",
        index=False,
    )

    result_rows = []

    for (degradation, metric), group in response.groupby(
        ["degradation", "metric"]
    ):
        means = (
            group.groupby("level")["value"]
            .mean()
            .reindex([1, 2, 3])
            .to_numpy()
        )

        diffs = np.diff(means)

        result_rows.append({
            "degradation": degradation,
            "metric": metric,
            "mean_level_1": means[0],
            "mean_level_2": means[1],
            "mean_level_3": means[2],
            "monotonic_increasing_or_decreasing": bool(
                np.all(diffs <= 0) or np.all(diffs >= 0)
            ),
            "strictly_decreasing": bool(
                np.all(diffs < 0)
            ),
        })

    summary = pd.DataFrame(result_rows)

    summary.to_csv(
        REPORTS / "quality_metric_response_summary_v2.csv",
        index=False,
    )

    print("FIXED_MASK_RESPONSE_SUMMARY")
    print(summary.to_string(index=False))
    print(
        "FIXED_MASK_NOTE=field_coverage_frame_percent is constant "
        "because the original mask is intentionally reused"
    )


def ungradable_report() -> None:
    """Compare ungradable Messidor-2 images with dataset distributions."""
    standard = pd.read_csv(
        REPORTS / "quality_metrics_standardised.csv"
    )

    names = [
        "20060411_58550_0200_PP.png",
        "IM002385.jpg",
        "IM003718.jpg",
        "IM004176.jpg",
    ]

    subset = standard[
        (standard.dataset == "Messidor-2")
        & standard.file_name.str.lower().str.endswith(
            tuple(name.lower() for name in names)
        )
    ].copy()

    cache_metrics = {}

    messidor_rows = standard[
        standard.dataset == "Messidor-2"
    ]

    for _, row in messidor_rows.iterrows():
        cache_metrics[row.file_name] = _cache_metric(
            cache_path("Messidor-2", row.file_name)
        )

    old_numeric = METRIC_NAMES

    new_numeric = [
        "sharpness_256",
        "field_coverage_frame_percent",
        "bbox_aspect_ratio",
        "circularity",
    ]

    records = []

    for _, row in subset.iterrows():
        cache = cache_metrics[row.file_name]

        record = {
            "file_name": row.file_name,
            "width": row.width,
            "height": row.height,
            "grade": row.grade,
        }

        for metric in old_numeric:
            record[metric] = row[metric]

        for metric in new_numeric:
            record[metric] = cache[metric]

        # Keep edge indicators as categorical observations.
        for field in EDGE_FIELDS:
            record[field] = cache[field]

        record["field_touches_edge"] = cache["field_touches_edge"]

        for metric in old_numeric:
            values = messidor_rows[metric].dropna().to_numpy()

            group = messidor_rows[
                (messidor_rows.width == row.width)
                & (messidor_rows.height == row.height)
            ][metric].dropna().to_numpy()

            record[f"{metric}_dataset_percentile"] = (
                100 * float(np.mean(values <= record[metric]))
                if len(values) else np.nan
            )

            record[f"{metric}_size_group_percentile"] = (
                100 * float(np.mean(group <= record[metric]))
                if len(group) else np.nan
            )

        for metric in new_numeric:
            values = np.array(
                [cache_metrics[f][metric] for f in cache_metrics],
                dtype=float,
            )

            matching_files = [
                f for f in cache_metrics
                if (
                    messidor_rows.loc[
                        messidor_rows.file_name == f,
                        "width",
                    ].iloc[0] == row.width
                    and
                    messidor_rows.loc[
                        messidor_rows.file_name == f,
                        "height",
                    ].iloc[0] == row.height
                )
            ]

            group_values = np.array(
                [cache_metrics[f][metric] for f in matching_files],
                dtype=float,
            )

            record[f"{metric}_dataset_percentile"] = (
                100 * float(np.mean(values <= record[metric]))
                if len(values) else np.nan
            )

            record[f"{metric}_size_group_percentile"] = (
                100 * float(np.mean(group_values <= record[metric]))
                if len(group_values) else np.nan
            )

        # Percentiles for Boolean edge indicators are intentionally omitted.
        for field in EDGE_FIELDS:
            record[f"{field}_dataset_percentile"] = np.nan
            record[f"{field}_size_group_percentile"] = np.nan

        records.append(record)

    output = pd.DataFrame(records)

    output.to_csv(
        REPORTS / "ungradable_messidor2_quality_values_v2.csv",
        index=False,
    )

    print("UNGRADABLE_V2")
    print(output.to_string(index=False))


def montages() -> None:
    """Generate the existing contrast review montage."""
    metrics = pd.read_csv(
        REPORTS / "quality_metrics_standardised.csv"
    )

    extra = []

    for _, row in metrics[
        metrics.dataset.isin(["APTOS", "Messidor-2"])
    ].iterrows():

        cache = _cache_metric(
            cache_path(row.dataset, row.file_name)
        )

        extra.append({
            "dataset": row.dataset,
            "file_name": row.file_name,
            "sharpness_256": cache["sharpness_256"],
        })

    data = metrics.merge(
        pd.DataFrame(extra),
        on=["dataset", "file_name"],
    )

    metric_names = ["contrast_p5_p95"]

    V2_REVIEW.mkdir(
        parents=True,
        exist_ok=True,
    )

    for metric in metric_names:
        sorted_data = (
            data[
                data.dataset.isin(["APTOS", "Messidor-2"])
            ]
            .sort_values(metric)
            .reset_index(drop=True)
        )

        if sorted_data.empty:
            continue

        indices = sorted(set(
            list(range(min(5, len(sorted_data))))
            + list(range(max(0, len(sorted_data) - 5), len(sorted_data)))
            + [
                round(i * (len(sorted_data) - 1) / 20)
                for i in range(1, 20)
            ]
        ))

        chosen = sorted_data.iloc[indices]

        fig, axes = plt.subplots(
            5,
            6,
            figsize=(18, 16),
        )

        for ax, (_, row) in zip(axes.flat, chosen.iterrows()):
            with np.load(
                cache_path(row.dataset, row.file_name)
            ) as z:
                ax.imshow(
                    z["gray512"],
                    cmap="gray",
                )

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
            f"Quality review v2: {metric} (no threshold selected)"
        )

        fig.tight_layout()

        output = V2_REVIEW / f"montage_{metric}.png"

        fig.savefig(
            output,
            dpi=150,
        )

        plt.close(fig)

        print(f"MONTAGE_FILE={output}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "step",
        choices=(
            "cache",
            "correlations",
            "stratified",
            "diagnostics",
            "ungradable",
            "montages",
            "all",
        ),
    )

    args = parser.parse_args()

    if args.step in ("cache", "all"):
        build_cache()

    if args.step in ("correlations", "all"):
        full_correlations()

    if args.step in ("stratified", "all"):
        stratified()

    if args.step in ("diagnostics", "all"):
        fixed_mask_diagnostics()

    if args.step in ("ungradable", "all"):
        ungradable_report()

    if args.step in ("montages", "all"):
        montages()


if __name__ == "__main__":
    main()
