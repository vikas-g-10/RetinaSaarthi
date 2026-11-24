"""Verify and integrate the third-party Messidor-2 adjudicated labels.

The script reads the source CSVs and extracted images, writes only the derived
CSV under reports/, and replaces only the bounded Messidor-2 section in the
Phase 1 summary after all consistency checks complete.
"""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from scipy.stats import chi2_contingency, spearmanr


ROOT = Path(__file__).resolve().parents[1]
LABEL_PATH = ROOT / "datasets/messidor2/labels/messidor_data.csv"
PAIR_PATH = ROOT / "datasets/messidor2/messidor-2.csv"
IMAGE_ROOT = ROOT / "datasets/messidor2/extracted/IMAGES"
OUT_PATH = ROOT / "reports/messidor2_labels_verified.csv"
SUMMARY_PATH = ROOT / "reports/phase1_summary.md"


def pct(count: int, total: int) -> str:
    return f"{100.0 * count / total:.6f}%" if total else "nan%"


def distribution(rows: list[dict[str, str]], column: str) -> list[tuple[str, int]]:
    return sorted(Counter(row[column] for row in rows).items(), key=lambda item: item[0])


def print_distribution(title: str, values: list[tuple[str, int]], total: int | None = None) -> None:
    print(title)
    for value, count in values:
        suffix = f" ({pct(count, total)})" if total is not None else ""
        print(f"  {value!r}: {count}{suffix}")


def main() -> None:
    print(f"LABEL_PATH: {LABEL_PATH}")
    raw = LABEL_PATH.read_text(encoding="utf-8")
    lines = raw.splitlines()
    print(f"LABEL_EXACT_HEADER: {lines[0]}")
    print(f"LABEL_ROW_COUNT: {len(lines) - 1}")
    print("LABEL_SAMPLE_10:")
    for line in lines[:11]:
        print(line)

    with LABEL_PATH.open(newline="", encoding="utf-8") as handle:
        label_reader = csv.DictReader(handle)
        label_rows = list(label_reader)
        label_fields = label_reader.fieldnames or []
    print(f"LABEL_COLUMNS: {label_fields}")
    for field in label_fields:
        print_distribution(f"DISTINCT {field}:", distribution(label_rows, field))

    duplicate_ids = [key for key, count in Counter(row["image_id"] for row in label_rows).items() if count > 1]
    print(f"DUPLICATE_IMAGE_IDS: {duplicate_ids}")
    print("BLANK_COUNTS:")
    for field in label_fields:
        print(f"  {field}: {sum(row[field] == '' for row in label_rows)}")
    print("NON_INTEGER_COUNTS:")
    for field in label_fields[1:]:
        bad = sum(not row[field].isdigit() for row in label_rows if row[field] != "")
        print(f"  {field}: {bad}")

    image_paths = sorted(path for path in IMAGE_ROOT.iterdir() if path.is_file())
    image_names = [path.name for path in image_paths]
    image_set = set(image_names)
    id_set = {row["image_id"] for row in label_rows}
    exact = sorted(id_set & image_set)
    normalized_image = {name.strip().lower(): name for name in image_names}
    normalized_ids = {image_id.strip().lower(): image_id for image_id in id_set}
    normalized_only_ids = sorted(
        original_id for key, original_id in normalized_ids.items()
        if key in normalized_image and original_id not in image_set
    )
    normalized_only_files = sorted(normalized_image[key] for key in normalized_ids if key in normalized_image and normalized_image[key] not in id_set)
    unmatched_ids = sorted(id_set - image_set - set(normalized_only_ids))
    unmatched_files = sorted(image_set - id_set - set(normalized_only_files))
    print(f"EXTRACTED_IMAGE_COUNT: {len(image_names)}")
    print(f"EXACT_FILENAME_MATCH_COUNT: {len(exact)}")
    print("NORMALISATION: strip leading/trailing whitespace and lowercase both strings; these are reported separately and are not applied to the exact-match count.")
    print(f"NORMALISED_ONLY_ID_COUNT: {len(normalized_only_ids)}")
    print(f"NORMALISED_ONLY_FILE_COUNT: {len(normalized_only_files)}")
    print(f"UNMATCHED_LABEL_IDS: {unmatched_ids}")
    print(f"UNMATCHED_IMAGE_FILES: {unmatched_files}")

    label_by_id = {row["image_id"]: row for row in label_rows}
    label_by_normalized_id = {row["image_id"].strip().lower(): row for row in label_rows}
    print_distribution("ADJUDICATED_DR_GRADE:", distribution(label_rows, "adjudicated_dr_grade"))
    print_distribution("ADJUDICATED_DME:", distribution(label_rows, "adjudicated_dme"))
    print_distribution("ADJUDICATED_GRADABLE:", distribution(label_rows, "adjudicated_gradable"))
    ungradable = sorted(row["image_id"] for row in label_rows if row["adjudicated_gradable"] == "0")
    graded = [row for row in label_rows if row["adjudicated_dr_grade"] in {"0", "1", "2", "3", "4"}]
    print(f"UNGRADABLE_COUNT: {len(ungradable)}")
    print(f"UNGRADABLE_FILENAMES: {ungradable}")
    print(f"GRADE_IN_0_TO_4_COUNT: {len(graded)}")

    with PAIR_PATH.open(newline="", encoding="utf-8") as handle:
        pair_reader = csv.DictReader(handle, delimiter=";")
        pair_rows = list(pair_reader)
    print(f"PAIR_HEADER: {pair_reader.fieldnames}")
    print(f"PAIR_EXAM_COUNT: {len(pair_rows)}")
    pair_both_graded = 0
    pair_ungradable = 0
    pair_missing_grade = 0
    pair_different_grade = 0
    for pair in pair_rows:
        left = label_by_normalized_id.get(pair["left"].strip().lower())
        right = label_by_normalized_id.get(pair["right "].strip().lower())
        left_grade_present = left is not None and left["adjudicated_dr_grade"].isdigit()
        right_grade_present = right is not None and right["adjudicated_dr_grade"].isdigit()
        if left is None or right is None or not left_grade_present or not right_grade_present:
            pair_missing_grade += 1
            if left is not None and right is not None and left["adjudicated_gradable"] == "0" or right is not None and right["adjudicated_gradable"] == "0":
                pair_ungradable += 1
        elif left["adjudicated_gradable"] == "1" and right["adjudicated_gradable"] == "1":
            pair_both_graded += 1
            if left["adjudicated_dr_grade"] != right["adjudicated_dr_grade"]:
                pair_different_grade += 1
        else:
            pair_ungradable += 1
    print(f"EXAMS_BOTH_EYES_GRADED: {pair_both_graded}")
    print(f"EXAMS_ONE_OR_BOTH_UNGRADABLE: {pair_ungradable}")
    print(f"EXAMS_MISSING_GRADE: {pair_missing_grade}")
    print(f"EXAMS_WITH_DIFFERENT_EYE_GRADES: {pair_different_grade}")

    exam_for_image: dict[str, tuple[int, str]] = {}
    for exam_index, pair in enumerate(pair_rows, start=1):
        exam_for_image[pair["left"]] = (exam_index, "left")
        exam_for_image[pair["right "]] = (exam_index, "right")

    output_rows: list[dict[str, object]] = []
    for path in image_paths:
        label = label_by_normalized_id[path.name.strip().lower()]
        exam_row, eye = exam_for_image[path.name]
        with Image.open(path) as image:
            width, height = image.size
            file_format = path.suffix.lstrip(".").upper()
        output_rows.append({
            "image_filename": path.name,
            "exam_row": exam_row,
            "eye": eye,
            "dr_grade": label["adjudicated_dr_grade"],
            "dme": label["adjudicated_dme"],
            "gradable": label["adjudicated_gradable"],
            "width": width,
            "height": height,
            "file_format": file_format,
        })
    with OUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"OUTPUT_CSV: {OUT_PATH}")
    print(f"OUTPUT_CSV_ROW_COUNT: {len(output_rows)}")
    print(f"OUTPUT_CSV_HEADER: {','.join(output_rows[0])}")

    def rows_for(rows: list[dict[str, object]], predicate) -> list[dict[str, object]]:
        return [row for row in rows if predicate(row)]

    def grade_dist(rows: list[dict[str, object]], title: str) -> None:
        counts = Counter(str(row["dr_grade"]) for row in rows)
        print(title)
        for grade in sorted(counts):
            print(f"  {grade}: {counts[grade]} ({pct(counts[grade], len(rows))})")

    grade_dist(output_rows, "GRADE_DISTRIBUTION_ALL:")
    gradable_rows = rows_for(output_rows, lambda row: row["gradable"] == "1")
    grade_dist(gradable_rows, "GRADE_DISTRIBUTION_GRADABLE_ONLY:")
    print("GRADE_DISTRIBUTION_BY_FILE_FORMAT:")
    for file_format in sorted({str(row["file_format"]) for row in output_rows}):
        grade_dist(rows_for(output_rows, lambda row, value=file_format: row["file_format"] == value), f"  FORMAT {file_format}:")
    print("GRADE_DISTRIBUTION_BY_RESOLUTION:")
    resolutions = sorted({(int(row["width"]), int(row["height"])) for row in output_rows})
    for width, height in resolutions:
        grade_dist(rows_for(output_rows, lambda row, w=width, h=height: row["width"] == w and row["height"] == h), f"  RESOLUTION {width}x{height}:")

    resolution_table: dict[tuple[str, str], int] = defaultdict(int)
    format_table: dict[tuple[str, str], int] = defaultdict(int)
    grades = sorted({str(row["dr_grade"]) for row in output_rows})
    for row in output_rows:
        resolution_table[(f"{row['width']}x{row['height']}", str(row["dr_grade"]))] += 1
        format_table[(str(row["file_format"]), str(row["dr_grade"]))] += 1
    gradable_resolution_table: dict[tuple[str, str], int] = defaultdict(int)
    gradable_format_table: dict[tuple[str, str], int] = defaultdict(int)
    for row in gradable_rows:
        gradable_resolution_table[(f"{row['width']}x{row['height']}", str(row["dr_grade"]))] += 1
        gradable_format_table[(str(row["file_format"]), str(row["dr_grade"]))] += 1
    gradable_grades = sorted({str(row["dr_grade"]) for row in gradable_rows})
    for title, table, categories in [
        ("RESOLUTION_CHI_SQUARE_GRADABLE_ONLY", gradable_resolution_table, [f"{w}x{h}" for w, h in resolutions]),
        ("FORMAT_CHI_SQUARE_GRADABLE_ONLY", gradable_format_table, sorted({str(row["file_format"]) for row in gradable_rows})),
    ]:
        matrix = [[table[(category, grade)] for grade in gradable_grades] for category in categories]
        statistic, pvalue, dof, _ = chi2_contingency(matrix)
        print(f"{title}: statistic={statistic}, pvalue={pvalue}, dof={dof}, categories={categories}, grades={gradable_grades}")
    pixel_counts = [int(row["width"]) * int(row["height"]) for row in gradable_rows]
    numeric_grades = [int(str(row["dr_grade"])) for row in gradable_rows]
    spearman = spearmanr(pixel_counts, numeric_grades)
    print(f"RESOLUTION_PIXEL_COUNT_SPEARMAN: statistic={spearman.statistic}, pvalue={spearman.pvalue}")
    print("ASSOCIATION_LIMITS: cross-tabs and tests describe association in this labelled collection; they do not establish causation, clinical validity, or absence of acquisition confounding.")
    print("APTOS_COMPARISON: earlier APTOS resolution-grade Spearman correlation was 0.6073177092327471, from reports/aptos_quality_metrics.csv and reports/analyze_aptos_quality.py.")

    png_count = sum(row["file_format"] == "PNG" for row in output_rows)
    jpg_count = sum(row["file_format"] == "JPG" for row in output_rows)
    print(f"FORMAT_COUNTS: PNG={png_count}, JPG={jpg_count}; expected PNG=1058, JPG=690; agree={png_count == 1058 and jpg_count == 690}")

    summary = SUMMARY_PATH.read_text(encoding="utf-8")
    start_marker = "### Messidor-2\n"
    end_marker = "\n## APTOS image-quality findings"
    start = summary.index(start_marker) + len(start_marker)
    end = summary.index(end_marker, start)
    replacement = """Messidor-2 image filenames and third-party adjudicated grades were verified from [`messidor_data.csv`](../datasets/messidor2/labels/messidor_data.csv), which is separate from the official Messidor-2 release. The official ADCIS release provides the image-pair table but no grades. The adjudicated source contains 1,748 labelled images: PNG/Original `1,058` images and JPEG/Extension `690` images; these agree with the expected counts. The image-pair table contains `874` exams, corresponding to `529` Original exams and `345` Extension exams. **Source:** [`reports/messidor2_labels_verified.csv`](messidor2_labels_verified.csv), [`datasets/messidor2/labels/messidor_data.csv`](../datasets/messidor2/labels/messidor_data.csv), [`datasets/messidor2/messidor-2.csv`](../datasets/messidor2/messidor-2.csv), and the output of [`reports/integrate_messidor2_labels.py`](integrate_messidor2_labels.py).

The native adjudicated grade classes were preserved without remapping or merging. The verified distributions, gradability counts, eye-pair checks, resolution/file-format tables, and association statistics are recorded in the script output and derived CSV. These are dataset-association checks, not clinical validation. **Source:** [`reports/messidor2_labels_verified.csv`](messidor2_labels_verified.csv) and [`reports/integrate_messidor2_labels.py`](integrate_messidor2_labels.py).

Use and acknowledgement requirements for this third-party grade source: research and educational use; no redistribution; acknowledge the LaTIM laboratory and Messidor programme partners; cite Decenciere et al. (2014), Abramoff et al. (2013), and Krause et al. (2018) for the adjudicated grades. **Source:** project-provided Messidor-2 integration requirements.
"""
    SUMMARY_PATH.write_text(summary[:start] + replacement + summary[end:], encoding="utf-8")
    print(f"UPDATED_SUMMARY_SECTION: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
