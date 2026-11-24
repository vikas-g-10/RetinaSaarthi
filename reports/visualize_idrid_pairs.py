from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np

base = Path("datasets/idrid")

seg_root = base / "segmentation/extracted/A. Segmentation/1. Original Images"
grading_root = base / "disease_grading/extracted/B. Disease Grading/1. Original Images"

selected_ids = [1, 5, 9, 43, 67, 71]

fig, axes = plt.subplots(len(selected_ids), 3, figsize=(15, 24))

for row, image_id in enumerate(selected_ids):

    number = f"{image_id:02d}"
    seg_path = None

    for split in ["a. Training Set", "b. Testing Set"]:
        candidate = seg_root / split / f"IDRiD_{number}.jpg"
        if candidate.exists():
            seg_path = candidate
            break

    grading_number = f"{image_id:03d}"
    grading_path = None

    for split in ["a. Training Set", "b. Testing Set"]:
        candidate = grading_root / split / f"IDRiD_{grading_number}.jpg"
        if candidate.exists():
            grading_path = candidate
            break

    if seg_path is None or grading_path is None:
        print(f"Missing pair: {image_id}")
        continue

    seg = cv2.cvtColor(
        cv2.imread(str(seg_path)), cv2.COLOR_BGR2RGB
    )

    grading = cv2.cvtColor(
        cv2.imread(str(grading_path)), cv2.COLOR_BGR2RGB
    )

    difference = np.abs(
        seg.astype(np.int16) - grading.astype(np.int16)
    ).astype(np.uint8)

    difference = np.clip(difference.astype(np.float32) * 3, 0, 255)
    difference = difference.astype(np.uint8)

    axes[row, 0].imshow(seg)
    axes[row, 0].set_title(f"ID {image_id:02d} - Segmentation")

    axes[row, 1].imshow(grading)
    axes[row, 1].set_title(f"ID {image_id:03d} - Disease Grading")

    axes[row, 2].imshow(difference)
    axes[row, 2].set_title("Amplified absolute difference (3x)")

    for col in range(3):
        axes[row, col].axis("off")

plt.tight_layout()

output = Path("reports/idrid_visual_comparison.png")
plt.savefig(output, dpi=150, bbox_inches="tight")
plt.close()

print("Saved:", output)
