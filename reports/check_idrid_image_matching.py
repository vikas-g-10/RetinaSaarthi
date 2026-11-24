from pathlib import Path
import cv2
import numpy as np

base = Path("datasets/idrid")

seg_dir = (
    base / "segmentation" / "extracted"
    / "A. Segmentation" / "1. Original Images"
)

grading_dir = (
    base / "disease_grading" / "extracted"
    / "B. Disease Grading" / "1. Original Images"
)

grading_splits = [
    grading_dir / "a. Training Set",
    grading_dir / "b. Testing Set",
]

grading_images = {}

for folder in grading_splits:
    for path in folder.glob("*.jpg"):
        grading_images[path.stem] = path

print("Disease-grading images indexed:", len(grading_images))

matched_ids = 0
exact_pixel_matches = 0
dimension_matches = 0

print("\nChecking corresponding numeric IDs...\n")

for seg_path in sorted(seg_dir.rglob("*.jpg")):
    # IDRiD_01 becomes IDRiD_001
    number = seg_path.stem.split("_")[-1]
    candidate_name = f"IDRiD_{int(number):03d}"

    if candidate_name not in grading_images:
        continue

    matched_ids += 1
    grade_path = grading_images[candidate_name]

    seg_img = cv2.imread(str(seg_path))
    grade_img = cv2.imread(str(grade_path))

    if seg_img is None or grade_img is None:
        print(candidate_name, "could not be decoded")
        continue

    same_dimensions = seg_img.shape == grade_img.shape

    if same_dimensions:
        dimension_matches += 1

        if np.array_equal(seg_img, grade_img):
            exact_pixel_matches += 1

    print(
        f"{seg_path.name} <-> {candidate_name}: "
        f"same dimensions={same_dimensions}, "
        f"exact pixels={same_dimensions and np.array_equal(seg_img, grade_img)}"
    )

print("\n" + "=" * 55)
print("Corresponding numeric IDs:", matched_ids)
print("Same dimensions:", dimension_matches)
print("Exact decoded pixel matches:", exact_pixel_matches)
