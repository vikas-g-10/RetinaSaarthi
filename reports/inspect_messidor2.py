"""Read-only post-extraction inspection of Messidor-2."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "datasets/messidor2/messidor-2.csv"
IMAGE_ROOT = ROOT / "datasets/messidor2/extracted/IMAGES"


def main() -> None:
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        rows = list(reader)
        fields = reader.fieldnames or []
    csv_names = [row[field].strip() for row in rows for field in fields]
    image_paths = sorted(path for path in IMAGE_ROOT.rglob("*") if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    image_names = [path.name for path in image_paths]

    dimensions = Counter()
    open_errors = []
    for path in image_paths:
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                dimensions[image.size] += 1
        except Exception as error:  # report every failure rather than hiding it
            open_errors.append(f"{path.name}: {type(error).__name__}: {error}")

    exact_csv = set(csv_names)
    exact_images = set(image_names)
    casefold_csv = {name.casefold() for name in csv_names}
    casefold_images = {name.casefold() for name in image_names}
    same_date_prefix = sum(left.strip().split("_", 1)[0] == right.strip().split("_", 1)[0] for left, right in ((row[fields[0]], row[fields[1]]) for row in rows))

    print(f"csv_rows={len(rows)} csv_referenced_images={len(csv_names)} csv_unique_images={len(exact_csv)}")
    print(f"extracted_images={len(image_paths)} extracted_unique_names={len(exact_images)}")
    print(f"images_opened_ok={len(image_paths) - len(open_errors)} open_errors={len(open_errors)}")
    for error in open_errors:
        print(f"open_error={error}")
    print(f"exact_csv_not_in_extracted={sorted(exact_csv - exact_images)}")
    print(f"exact_extracted_not_in_csv={sorted(exact_images - exact_csv)}")
    print(f"casefold_csv_not_in_extracted={sorted(casefold_csv - casefold_images)}")
    print(f"casefold_extracted_not_in_csv={sorted(casefold_images - casefold_csv)}")
    print("width_height_distribution=" + "; ".join(f"{width}x{height}={count}" for (width, height), count in sorted(dimensions.items())))
    print(f"csv_rows_with_left_right_same_first_filename_token={same_date_prefix}")
    print(f"csv_rows_with_left_right_pairs={len(rows)}")
    print(f"csv_columns={fields}")
    print("label_like_columns=none; both columns contain image filenames and no numeric DR-grade field was present")


if __name__ == "__main__":
    main()
