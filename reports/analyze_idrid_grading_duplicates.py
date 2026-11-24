"""Tight duplicate analysis for the IDRiD disease-grading images.

Report-only: reads source labels/images and writes only under reports/. It
audits the preliminary connected-component groups, then verifies candidates
with green-channel correlation plus 1024-pixel-wide MAE and SSIM checks.
Final groups are maximal cliques: every pair within a reported group passes.
"""

from __future__ import annotations

import csv
import itertools
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageOps
from skimage.metrics import structural_similarity


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets/idrid/disease_grading/extracted/B. Disease Grading"
REPORT_ROOT = ROOT / "reports"
OUTPUT_CSV = REPORT_ROOT / "idrid_grading_duplicates.csv"
PRELIMINARY_SUMMARY_CSV = REPORT_ROOT / "idrid_grading_preliminary_group_summary.csv"
SHEETS_ROOT = REPORT_ROOT / "idrid_dup_sheets"

CORRELATION_THRESHOLD = 0.995
MAE_THRESHOLD = 5.0
SSIM_THRESHOLD = 0.90
CORRELATION_WIDTH = 256
PIXEL_WIDTH = 1024

LABEL_FILES = {
    "train": DATA_ROOT / "2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv",
    "test": DATA_ROOT / "2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv",
}
IMAGE_DIRS = {
    "train": DATA_ROOT / "1. Original Images/a. Training Set",
    "test": DATA_ROOT / "1. Original Images/b. Testing Set",
}


def read_labels(path: Path) -> dict[str, tuple[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle)
        required = {"Image name", "Retinopathy grade", "Risk of macular edema "}
        if not required.issubset(rows.fieldnames or set()):
            raise ValueError(f"Unexpected columns in {path}: {rows.fieldnames}")
        result = {}
        for row in rows:
            stem = row["Image name"].strip()
            if stem:
                result[stem] = (row["Retinopathy grade"].strip(), row["Risk of macular edema "].strip())
        return result


def source_records() -> list[dict[str, object]]:
    result = []
    for split in ("train", "test"):
        labels = read_labels(LABEL_FILES[split])
        for path in sorted(IMAGE_DIRS[split].glob("*.jpg")):
            if path.stem not in labels:
                raise ValueError(f"No label row for {split}/{path.name}")
            grade, edema = labels[path.stem]
            result.append({"split": split, "filename": path.name, "path": path, "grade": grade, "edema": edema})
    return result


def legacy_feature(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        values = np.asarray(ImageOps.grayscale(image).resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32).ravel()
    values -= values.mean()
    return values / np.linalg.norm(values)


def green_feature(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read {path}")
    width = CORRELATION_WIDTH
    height = round(image.shape[0] * width / image.shape[1])
    green = cv2.resize(image[:, :, 1], (width, height), interpolation=cv2.INTER_AREA)
    green = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(green)
    green = cv2.GaussianBlur(green, (5, 5), 0)
    values = green.astype(np.float32).ravel()
    values -= values.mean()
    return values / np.linalg.norm(values)


def pixel_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not read {path}")
    height = round(image.shape[0] * PIXEL_WIDTH / image.shape[1])
    return cv2.resize(image, (PIXEL_WIDTH, height), interpolation=cv2.INTER_AREA)


def all_pair_scores(features: list[np.ndarray]) -> list[tuple[float, int, int]]:
    matrix = np.stack(features)
    scores = np.clip(matrix @ matrix.T, -1.0, 1.0)
    return [(float(scores[left, right]), left, right) for left, right in itertools.combinations(range(len(features)), 2)]


def component_groups(pair_scores: list[tuple[float, int, int]], count: int, threshold: float) -> list[list[int]]:
    parent = list(range(count))

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    for score, left, right in pair_scores:
        if score >= threshold:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root
    grouped = defaultdict(list)
    for index in range(count):
        grouped[find(index)].append(index)
    return sorted((members for members in grouped.values() if len(members) > 1), key=lambda group: (group[0], len(group)))


def maximal_cliques(adjacency: list[set[int]]) -> list[list[int]]:
    found: list[tuple[int, ...]] = []

    def bron_kerbosch(current: set[int], candidates: set[int], excluded: set[int]) -> None:
        if not candidates and not excluded:
            if len(current) > 1:
                found.append(tuple(sorted(current)))
            return
        pivot_pool = candidates | excluded
        pivot = max(pivot_pool, key=lambda item: len(candidates & adjacency[item]), default=None)
        options = candidates - (adjacency[pivot] if pivot is not None else set())
        for item in list(options):
            bron_kerbosch(current | {item}, candidates & adjacency[item], excluded & adjacency[item])
            candidates.remove(item)
            excluded.add(item)

    bron_kerbosch(set(), set(range(len(adjacency))), set())
    return [list(group) for group in sorted(set(found), key=lambda group: (group[0], len(group), group))]


def pairwise_metrics(records: list[dict[str, object]], correlation_scores: list[tuple[float, int, int]]) -> list[dict[str, object]]:
    pixel_cache: dict[int, np.ndarray] = {}
    metrics = []
    for correlation, left, right in correlation_scores:
        if correlation < CORRELATION_THRESHOLD:
            continue
        pixel_cache.setdefault(left, pixel_image(records[left]["path"]))
        pixel_cache.setdefault(right, pixel_image(records[right]["path"]))
        left_image, right_image = pixel_cache[left], pixel_cache[right]
        mae = float(np.mean(np.abs(left_image.astype(np.float32) - right_image.astype(np.float32))))
        ssim = float(structural_similarity(left_image, right_image, data_range=255))
        metrics.append({"correlation": correlation, "mae": mae, "ssim": ssim, "left": left, "right": right,
                        "passes": mae <= MAE_THRESHOLD and ssim >= SSIM_THRESHOLD})
    return metrics


def disagreement(records: list[dict[str, object]], group: list[int]) -> bool:
    return len({(records[index]["grade"], records[index]["edema"]) for index in group}) > 1


def write_preliminary_summary(records: list[dict[str, object]], groups: list[list[int]], scores: list[tuple[float, int, int]]) -> None:
    fields = ["preliminary_group_id", "group_size", "qualifying_pairs", "possible_pairs", "full_clique", "members"]
    with PRELIMINARY_SUMMARY_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group_id, group in enumerate(groups, 1):
            member_set = set(group)
            qualifying = sum(score >= 0.995 and left in member_set and right in member_set for score, left, right in scores)
            possible = len(group) * (len(group) - 1) // 2
            members = "; ".join(f"({records[i]['split']}, {records[i]['filename']})" for i in group)
            writer.writerow({"preliminary_group_id": group_id, "group_size": len(group), "qualifying_pairs": qualifying,
                             "possible_pairs": possible, "full_clique": qualifying == possible, "members": members})
            print(f"preliminary_group_{group_id:03d}: qualifying_pairs={qualifying} possible_pairs={possible} full_clique={qualifying == possible}")


def write_verified_csv(records: list[dict[str, object]], groups: list[list[int]], metrics: list[dict[str, object]]) -> None:
    best = defaultdict(float)
    for metric in metrics:
        if metric["passes"]:
            best[metric["left"]] = max(best[metric["left"]], metric["correlation"])
            best[metric["right"]] = max(best[metric["right"]], metric["correlation"])
    fields = ["group_id", "group_size", "split", "filename", "retinopathy_grade", "macular_edema_risk", "label_disagreement", "best_correlation"]
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for group_id, group in enumerate(groups, 1):
            disagree = disagreement(records, group)
            for index in sorted(group, key=lambda item: (records[item]["split"], records[item]["filename"])):
                writer.writerow({"group_id": group_id, "group_size": len(group), "split": records[index]["split"],
                                 "filename": records[index]["filename"], "retinopathy_grade": records[index]["grade"],
                                 "macular_edema_risk": records[index]["edema"], "label_disagreement": disagree,
                                 "best_correlation": f"{best[index]:.8f}"})


def make_sheet(records: list[dict[str, object]], pairs: list[tuple[int, int]], name: str) -> Path:
    rows = []
    for left, right in pairs:
        with Image.open(records[left]["path"]) as source:
            first = ImageOps.exif_transpose(source).convert("RGB").resize((500, 500), Image.Resampling.LANCZOS)
        with Image.open(records[right]["path"]) as source:
            second = ImageOps.exif_transpose(source).convert("RGB").resize((500, 500), Image.Resampling.LANCZOS)
        diff = ImageOps.autocontrast(ImageOps.grayscale(ImageChops.difference(first, second)))
        diff = ImageEnhance.Contrast(diff).enhance(4.0)
        diff_rgb = Image.new("RGB", diff.size)
        diff_rgb.paste(diff)
        row = Image.new("RGB", (1560, 570), "white")
        row.paste(first, (0, 10))
        row.paste(second, (520, 10))
        row.paste(diff_rgb, (1040, 10))
        draw = ImageDraw.Draw(row)
        draw.text((10, 520), f"reference/seg slot: {records[left]['split']}/{records[left]['filename']} DR={records[left]['grade']} ME={records[left]['edema']}", fill="black")
        draw.text((530, 520), f"grading candidate: {records[right]['split']}/{records[right]['filename']} DR={records[right]['grade']} ME={records[right]['edema']}", fill="black")
        draw.text((1050, 520), "absolute difference", fill="black")
        rows.append(row)
    sheet = Image.new("RGB", (1560, 570 * len(rows)), "white")
    for row_number, row in enumerate(rows):
        sheet.paste(row, (0, row_number * 570))
    path = SHEETS_ROOT / name
    sheet.save(path, quality=92)
    return path


def main() -> None:
    records = source_records()
    legacy_scores = all_pair_scores([legacy_feature(record["path"]) for record in records])
    preliminary_groups = component_groups(legacy_scores, len(records), 0.995)
    write_preliminary_summary(records, preliminary_groups, legacy_scores)

    correlation_scores = all_pair_scores([green_feature(record["path"]) for record in records])
    metrics = pairwise_metrics(records, correlation_scores)
    adjacency = [set() for _ in records]
    for metric in metrics:
        if metric["passes"]:
            adjacency[metric["left"]].add(metric["right"])
            adjacency[metric["right"]].add(metric["left"])
    verified_groups = maximal_cliques(adjacency)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    SHEETS_ROOT.mkdir(parents=True, exist_ok=True)
    write_verified_csv(records, verified_groups, metrics)

    preliminary_largest = max(preliminary_groups, key=len)
    by_key = {(record["split"], record["filename"]): index for index, record in enumerate(records)}
    target_pairs = [
        ("train_073_vs_train_375.jpg", by_key[("train", "IDRiD_073.jpg")], by_key[("train", "IDRiD_375.jpg")]),
        ("train_028_vs_train_327.jpg", by_key[("train", "IDRiD_028.jpg")], by_key[("train", "IDRiD_327.jpg")]),
    ]
    sheets = [make_sheet(records, [(left, right)], name) for name, left, right in target_pairs]
    largest_pairs = [(preliminary_largest[0], member) for member in preliminary_largest[1:]]
    sheets.append(make_sheet(records, largest_pairs, "preliminary_largest_group_001.jpg"))

    all_scores = np.asarray([score for score, _, _ in correlation_scores])
    candidate_scores = np.asarray([metric["correlation"] for metric in metrics])
    print(f"records={len(records)} train=413 test=103 all_pairs={len(correlation_scores)}")
    print(f"method=green_channel_width_{CORRELATION_WIDTH}_clahe_clip2_tile8_blur5 correlation_threshold={CORRELATION_THRESHOLD:.6f} mae_threshold={MAE_THRESHOLD:.1f} ssim_threshold={SSIM_THRESHOLD:.2f}")
    print("correlation_quantiles=" + ", ".join(f"q={q}:{value:.8f}" for q, value in zip((50, 90, 99, 99.9, 100), np.quantile(all_scores, [0.5, 0.9, 0.99, 0.999, 1.0]))))
    edges = [0.90, 0.95, 0.98, 0.99, 0.995, 0.997, 0.999, 1.000001]
    counts = np.histogram(all_scores, bins=edges)[0]
    print("correlation_histogram=" + "; ".join(f"[{edges[i]:.3f},{edges[i + 1]:.3f})={count}" for i, count in enumerate(counts)))
    print(f"correlation_candidates={len(metrics)} candidate_score_range=({candidate_scores.min():.8f},{candidate_scores.max():.8f})")
    print("candidate_pixel_metrics_sorted=" + "; ".join(f"corr={m['correlation']:.8f},mae={m['mae']:.4f},ssim={m['ssim']:.8f},pass={m['passes']}" for m in sorted(metrics, key=lambda item: item["mae"])))
    print(f"verified_duplicate_groups={len(verified_groups)} group_sizes={[len(group) for group in verified_groups]} cross_train_test={sum(any(records[i]['split'] == 'train' for i in group) and any(records[i]['split'] == 'test' for i in group) for group in verified_groups)} label_disagreement={sum(disagreement(records, group) for group in verified_groups)}")
    print("target_pair_results=")
    for name, left, right in target_pairs:
        result = next((metric for metric in metrics if {metric["left"], metric["right"]} == {left, right}), None)
        print(f"{name}: " + (f"corr={result['correlation']:.8f} mae={result['mae']:.4f} ssim={result['ssim']:.8f} pass={result['passes']}" if result else "correlation_below_candidate_threshold"))
    print(f"verified_csv={OUTPUT_CSV} lines={sum(1 for _ in OUTPUT_CSV.open(encoding='utf-8'))}")
    print(f"verified_csv_header={OUTPUT_CSV.read_text(encoding='utf-8').splitlines()[0]}")
    for sheet in sheets:
        with Image.open(sheet) as image:
            print(f"sheet={sheet.name} pixels={image.width}x{image.height}")


if __name__ == "__main__":
    main()
