
from pathlib import Path
import re
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity, peak_signal_noise_ratio

base = Path("datasets/idrid")

seg_root = (
    base / "segmentation/extracted/"
    "A. Segmentation/1. Original Images"
)

grading_root = (
    base / "disease_grading/extracted/"
    "B. Disease Grading/1. Original Images"
)

splits = {
    "a. Training Set": "training",
    "b. Testing Set": "testing",
}


def get_numeric_id(filename):
    match = re.search(r"IDRiD_(\d+)", filename)
    return int(match.group(1)) if match else None


results = []

for split_folder, split_name in splits.items():

    seg_folder = seg_root / split_folder
    grading_folder = grading_root / split_folder

    grading_images = {}

    for path in grading_folder.glob("*.jpg"):
        numeric_id = get_numeric_id(path.name)

        if numeric_id is not None:
            grading_images[numeric_id] = path

    for seg_path in sorted(seg_folder.glob("*.jpg")):

        numeric_id = get_numeric_id(seg_path.name)

        if numeric_id not in grading_images:
            print(f"NO MATCH: {split_name} - {seg_path.name}")
            continue

        grading_path = grading_images[numeric_id]

        seg_img = cv2.imread(str(seg_path))
        grading_img = cv2.imread(str(grading_path))

        if seg_img is None or grading_img is None:
            print(f"UNREADABLE: {split_name} - ID {numeric_id}")
            continue

        if seg_img.shape != grading_img.shape:
            print(f"SIZE MISMATCH: {split_name} - ID {numeric_id}")
            continue

        seg_gray = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
        grading_gray = cv2.cvtColor(grading_img, cv2.COLOR_BGR2GRAY)

        ssim = structural_similarity(
            seg_gray, grading_gray, data_range=255
        )

        psnr = peak_signal_noise_ratio(
            seg_gray, grading_gray, data_range=255
        )

        mae = np.mean(
            np.abs(
                seg_gray.astype(np.float32)
                - grading_gray.astype(np.float32)
            )
        )

        results.append({
            "split": split_name,
            "numeric_id": numeric_id,
            "segmentation_image": seg_path.name,
            "grading_image": grading_path.name,
            "ssim": ssim,
            "psnr": psnr,
            "mae": mae,
        })


df = pd.DataFrame(results)

output = Path("reports/idrid_image_similarity_corrected.csv")
df.to_csv(output, index=False)

print("\n" + "=" * 60)
print("CORRECTED IDRiD IMAGE SIMILARITY RESULTS")
print("=" * 60)

print("Successfully compared pairs:", len(df))

if not df.empty:

    print("\nPairs by split:")
    print(df.groupby("split").size())

    print("\nSSIM statistics:")
    print(df["ssim"].describe().round(4))

    print("\nPSNR statistics:")
    print(df["psnr"].describe().round(4))

    print("\nMAE statistics:")
    print(df["mae"].describe().round(4))

    print("\nSelected pairs: ID 01 and ID 09")
    print(
        df[df["numeric_id"].isin([1, 9])]
        .to_string(index=False)
    )

    print("\nResults saved to:", output)

else:
    print("No image pairs were successfully compared.")
