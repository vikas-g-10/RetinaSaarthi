"""Build the one-row-per-segmentation-image IDRiD eligibility report.

This script reads the existing segmentation files, grading labels, duplicate
report, and mapping report. It writes only reports/idrid_eligibility.csv and
does not alter source datasets or pairing decisions.
"""

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEG_ROOT = ROOT / "datasets/idrid/segmentation/extracted/A. Segmentation"
GRADE_ROOT = ROOT / "datasets/idrid/disease_grading/extracted/B. Disease Grading"
REPORT_ROOT = ROOT / "reports"
OUTPUT = REPORT_ROOT / "idrid_eligibility.csv"
MAPPING = REPORT_ROOT / "idrid_mapping/mapping.csv"
DUPLICATES = REPORT_ROOT / "idrid_grading_duplicates.csv"

MASK_TYPES = ("MA", "HE", "EX", "SE", "OD")
MASK_DIRS = {
    "train": {"MA": "1. Microaneurysms", "HE": "2. Haemorrhages", "EX": "3. Hard Exudates", "SE": "4. Soft Exudates", "OD": "5. Optic Disc"},
    "test": {"MA": "1. Microaneurysms", "HE": "2. Haemorrhages", "EX": "3. Hard Exudates", "SE": "4. Soft Exudates", "OD": "5. Optic Disc"},
}
ORIGINAL_DIRS = {
    "train": SEG_ROOT / "1. Original Images/a. Training Set",
    "test": SEG_ROOT / "1. Original Images/b. Testing Set",
}
GROUNDTRUTH_ROOTS = {
    "train": SEG_ROOT / "2. All Segmentation Groundtruths/a. Training Set",
    "test": SEG_ROOT / "2. All Segmentation Groundtruths/b. Testing Set",
}
LABEL_FILES = {
    "train": GRADE_ROOT / "2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv",
    "test": GRADE_ROOT / "2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv",
}

HELD_OUT = {
    ("train", "IDRiD_39.jpg"),
    ("train", "IDRiD_26.jpg"),
    ("train", "IDRiD_11.jpg"),
    ("train", "IDRiD_25.jpg"),
    ("test", "IDRiD_56.jpg"),
    ("test", "IDRiD_70.jpg"),
}
MASK_RE = re.compile(r"^(IDRiD_\d+)_([A-Z]{2})\.tif$")


def read_labels() -> dict[tuple[str, str], tuple[str, str]]:
    labels = {}
    for split, path in LABEL_FILES.items():
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            required = {"Image name", "Retinopathy grade", "Risk of macular edema "}
            if not required.issubset(reader.fieldnames or set()):
                raise ValueError(f"Unexpected grading columns in {path}: {reader.fieldnames}")
            for row in reader:
                stem = row["Image name"].strip()
                if stem:
                    labels[(split, stem)] = (row["Retinopathy grade"].strip(), row["Risk of macular edema "].strip())
    return labels


def read_mapping() -> list[dict[str, str]]:
    with MAPPING.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 81:
        raise ValueError(f"Expected 81 mapping rows, found {len(rows)}")
    keys = [(row["seg_split"], row["seg_file"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Mapping has duplicate segmentation keys")
    return rows


def read_duplicate_members() -> tuple[dict[tuple[str, str], bool], set[tuple[str, str]]]:
    with DUPLICATES.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    conflict = {}
    members = set()
    for row in rows:
        key = (row["split"], row["filename"])
        members.add(key)
        conflict[key] = row["label_disagreement"].strip().lower() == "true"
    return conflict, members


def check_masks() -> tuple[dict[tuple[str, str], dict[str, bool]], list[str], dict[str, list[str]]]:
    originals = {}
    for split, directory in ORIGINAL_DIRS.items():
        originals[split] = {path.name for path in directory.glob("*.jpg")}
    presence = {}
    missing = []
    examples = {}
    for split, root in GROUNDTRUTH_ROOTS.items():
        for mask_type in MASK_TYPES:
            directory = root / MASK_DIRS[split][mask_type]
            files = sorted(directory.glob("*.tif"))
            examples[f"{split}/{mask_type}"] = [path.name for path in files[:5]]
            for path in files:
                match = MASK_RE.match(path.name)
                if not match or match.group(2) != mask_type:
                    missing.append(f"unexpected mask filename: {path}")
                    continue
                stem = match.group(1)
                original = f"{stem}.jpg"
                if original not in originals[split]:
                    missing.append(f"{split}/{path.name} -> missing original {original}")
                key = (split, original)
                presence.setdefault(key, {})[mask_type] = True
    for split, names in originals.items():
        for filename in names:
            presence.setdefault((split, filename), {})
            for mask_type in MASK_TYPES:
                presence[(split, filename)].setdefault(mask_type, False)
    return presence, missing, examples


def grade_key(value: str) -> tuple[str, str]:
    split, filename = value.split("/", 1)
    return split, filename


def main() -> None:
    labels = read_labels()
    mapping = read_mapping()
    conflict_by_grade, duplicate_members = read_duplicate_members()
    mask_presence, missing_masks, examples = check_masks()
    print("mask_filename_pattern=IDRiD_<number>_<MA|HE|EX|SE|OD>.tif -> same-split IDRiD_<number>.jpg")
    for key, names in examples.items():
        print(f"mask_examples[{key}]={names}")
    print(f"mask_to_original_missing_count={len(missing_masks)}")
    for item in missing_masks:
        print(f"mask_to_original_missing={item}")
    if missing_masks:
        raise ValueError("Mask filename mapping contradiction; stopping before eligibility output")

    confident_partner_duplicates = []
    rows = []
    for source in mapping:
        seg_key = (source["seg_split"], source["seg_file"])
        grade_split = ""
        grade_file = ""
        grade = ""
        edema = ""
        notes = ""
        source_grade = f"{source['grade_split']}/{source['grade_file']}" if source["grade_split"] and source["grade_file"] else ""
        primary_key = grade_key(source_grade) if source_grade else None
        if primary_key and primary_key in duplicate_members and source["status"] == "confident":
            confident_partner_duplicates.append((seg_key, primary_key))

        if seg_key in HELD_OUT:
            status = "held_out_imperfect"
        elif seg_key == ("test", "IDRiD_62.jpg"):
            status = "tie_duplicate"
            source_grade = "train/IDRiD_073.jpg"
            notes = "Tie candidates train/IDRiD_073.jpg and train/IDRiD_375.jpg; labels agree (DR 2, ME 1)."
        elif seg_key == ("test", "IDRiD_69.jpg"):
            status = "tie_label_conflict"
            source_grade = "train/IDRiD_327.jpg"
            notes = "Tie candidates train/IDRiD_327.jpg (DR 3, ME 2) and train/IDRiD_028.jpg (DR 4, ME 2); DR grade left blank."
        elif source["status"] == "confident":
            status = "confident"
        else:
            status = "unpaired"
            notes = f"Source mapping status={source['status']}; no verified grading partner used."

        if source_grade and "/" in source_grade and status != "unpaired":
            grade_split, grade_file = grade_key(source_grade)
            grade = labels[(grade_split, grade_file.removesuffix(".jpg"))][0]
            edema = labels[(grade_split, grade_file.removesuffix(".jpg"))][1]
        if status == "tie_label_conflict":
            grade = ""
            edema = ""
        grade_label_conflict = status != "unpaired" and bool(primary_key and conflict_by_grade.get(primary_key, False))
        if seg_key == ("test", "IDRiD_69.jpg"):
            grade_label_conflict = True
        row = {
            "seg_split": source["seg_split"],
            "seg_file": source["seg_file"],
            "grade_split": grade_split,
            "grade_file": grade_file,
            "pairing_status": status,
            "dr_grade": grade,
            "edema_risk": edema,
            "has_MA": mask_presence[seg_key]["MA"],
            "has_HE": mask_presence[seg_key]["HE"],
            "has_EX": mask_presence[seg_key]["EX"],
            "has_SE": mask_presence[seg_key]["SE"],
            "has_OD": mask_presence[seg_key]["OD"],
            "grade_label_conflict": grade_label_conflict,
            "notes": (notes + " " if notes else "") + "Boolean has_* columns mean mask file present; False does not mean no lesion. OD is anatomical, not a DR lesion.",
        }
        rows.append(row)

    if confident_partner_duplicates:
        print(f"CONTRADICTION: confident pairings use duplicate grading images: {confident_partner_duplicates}")
        raise ValueError("Unexpected confident duplicate grading partner")
    if len(rows) != 81:
        raise ValueError(f"Expected 81 eligibility rows, found {len(rows)}")

    fields = ["seg_split", "seg_file", "grade_split", "grade_file", "pairing_status", "dr_grade", "edema_risk", "has_MA", "has_HE", "has_EX", "has_SE", "has_OD", "grade_label_conflict", "notes"]
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    eligible_statuses = {"confident", "tie_duplicate"}
    eligible_with_conflict = eligible_statuses | {"tie_label_conflict"}
    for label, statuses in (("without_tie_label_conflict", eligible_statuses), ("with_tie_label_conflict", eligible_with_conflict)):
        selected = [row for row in rows if row["pairing_status"] in statuses]
        print(f"eligible_{label}_images={len(selected)}")
        for mask_type in MASK_TYPES:
            print(f"eligible_{label}_{mask_type}={sum(row[f'has_{mask_type}'] for row in selected)}")
        print(f"eligible_{label}_dr_distribution={dict(sorted(Counter(row['dr_grade'] for row in selected if row['dr_grade']).items()))}")
        print(f"eligible_{label}_seg_split_counts={dict(sorted(Counter(row['seg_split'] for row in selected).items()))}")
        print(f"eligible_{label}_cross_train_test={sum(row['seg_split'] != row['grade_split'] for row in selected)}")
    print(f"confident_partner_duplicate_count={len(confident_partner_duplicates)}")
    print(f"output={OUTPUT} rows={len(rows)} lines={len(rows) + 1}")
    print(f"header={','.join(fields)}")


if __name__ == "__main__":
    main()
