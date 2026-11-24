"""Independent, read-only verification of idrid_eligibility.csv.

This verifier independently parses the eligibility CSV and the two raw grading
CSV files. It does not import or call build_idrid_eligibility.py.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ELIGIBILITY = ROOT / "reports/idrid_eligibility.csv"
LABEL_FILES = {
    "train": ROOT / "datasets/idrid/disease_grading/extracted/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv",
    "test": ROOT / "datasets/idrid/disease_grading/extracted/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv",
}
FIELDS = ["seg_split", "seg_file", "grade_split", "grade_file", "pairing_status", "dr_grade", "edema_risk", "has_MA", "has_HE", "has_EX", "has_SE", "has_OD", "grade_label_conflict", "notes"]


def raw_labels() -> dict[tuple[str, str], tuple[str, str]]:
    result = {}
    for split, path in LABEL_FILES.items():
        with path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                result[(split, row["Image name"].strip())] = (row["Retinopathy grade"].strip(), row["Risk of macular edema "].strip())
    return result


def csv_line(row: dict[str, str]) -> str:
    import io
    output = io.StringIO()
    csv.writer(output, lineterminator="").writerow([row[field] for field in FIELDS])
    return output.getvalue()


def main() -> None:
    labels = raw_labels()
    with ELIGIBILITY.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["pairing_status"] in {"confident", "tie_duplicate"}]
    rows.sort(key=lambda row: (row["seg_split"], row["seg_file"]))
    print(f"eligible_rows_selected={len(rows)}")

    counts = {"dr_grade": {"match": 0, "mismatch": 0, "missing": 0}, "edema_risk": {"match": 0, "mismatch": 0, "missing": 0}}
    mismatches = []
    for row in rows:
        key = (row["grade_split"], row["grade_file"].removesuffix(".jpg"))
        raw = labels.get(key)
        for field, raw_index in (("dr_grade", 0), ("edema_risk", 1)):
            if raw is None or not row[field]:
                counts[field]["missing"] += 1
            elif row[field] == raw[raw_index]:
                counts[field]["match"] += 1
            else:
                counts[field]["mismatch"] += 1
                mismatches.append((row, raw, field))
    print(f"dr_grade_counts={counts['dr_grade']}")
    print(f"edema_risk_counts={counts['edema_risk']}")
    print(f"mismatch_rows={len(mismatches)}")
    for row, raw, field in mismatches:
        print(f"mismatch field={field} eligibility={csv_line(row)} raw_grade={raw[0]} raw_edema={raw[1]}")

    rng = random.Random(20261109)
    sample = rng.sample(rows, 5)
    print("random_sample_seed=20261109")
    for row in sample:
        key = (row["grade_split"], row["grade_file"].removesuffix(".jpg"))
        raw = labels.get(key)
        print(f"eligibility={csv_line(row)} raw_dr_grade={raw[0] if raw else '<missing>'} raw_edema_risk={raw[1] if raw else '<missing>'}")


if __name__ == "__main__":
    main()
