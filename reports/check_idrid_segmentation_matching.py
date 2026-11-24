
"""Check all exact pixel matches between IDRiD segmentation and grading images."""

from pathlib import Path
from hashlib import sha256
from collections import defaultdict

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

SEG_ROOT = (
    ROOT
    / "datasets/idrid/segmentation/extracted/A. Segmentation"
    / "1. Original Images"
)

GRADE_ROOT = (
    ROOT
    / "datasets/idrid/disease_grading/extracted/B. Disease Grading"
    / "1. Original Images"
)


def image_hash(path):
    """Hash decoded RGB pixels, independent of JPEG metadata."""

    with Image.open(path) as image:
        image = image.convert("RGB")
        digest = sha256()
        digest.update(str(image.size).encode())
        digest.update(image.tobytes())

    return digest.hexdigest()


def collect_images(root):
    """Keep every file path, including duplicate images."""

    result = defaultdict(list)

    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            result[image_hash(path)].append(path)

    return result


def main():
    print("Reading segmentation images...")
    segmentation = collect_images(SEG_ROOT)

    print("Reading disease-grading images...")
    grading = collect_images(GRADE_ROOT)

    matches = []

    for digest, seg_paths in segmentation.items():
        if digest in grading:
            for seg_path in seg_paths:
                for grade_path in grading[digest]:
                    matches.append((seg_path, grade_path))

    segmentation_count = sum(len(paths) for paths in segmentation.values())
    grading_count = sum(len(paths) for paths in grading.values())

    print("\n========== MATCHING RESULTS ==========")
    print(f"Segmentation image files: {segmentation_count}")
    print(f"Unique segmentation pixel hashes: {len(segmentation)}")
    print(f"Grading image files: {grading_count}")
    print(f"Unique grading pixel hashes: {len(grading)}")
    print(f"Exact matching file pairs: {len(matches)}")

    for seg_path, grade_path in matches:
        print(f"\nSegmentation: {seg_path.relative_to(SEG_ROOT)}")
        print(f"Grading:      {grade_path.relative_to(GRADE_ROOT)}")

    unmatched = [
        path
        for digest, paths in segmentation.items()
        if digest not in grading
        for path in paths
    ]

    print(f"\nUnmatched segmentation image files: {len(unmatched)}")


if __name__ == "__main__":
    main()
