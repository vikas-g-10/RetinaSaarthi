
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]

IMAGE_DIR = (
    PROJECT_ROOT
    / "datasets/idrid/disease_grading/extracted/B. Disease Grading/"
      "1. Original Images/b. Testing Set"
)

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs/classification/combined_aptos_idrid_baseline/"
      "external_idrid/external_predictions.csv"
)

OUTPUT_PATH = PROJECT_ROOT / "reports/idrid_quality_metrics.csv"

predictions = pd.read_csv(PREDICTIONS_PATH)

results = []

print("Analyzing IDRiD image quality...")
print("Total prediction records:", len(predictions))

for _, row in tqdm(predictions.iterrows(), total=len(predictions)):
    image_id = str(row["image_id"])
    image_path = IMAGE_DIR / f"{image_id}.jpg"

    image = cv2.imread(str(image_path))

    if image is None:
        print("WARNING: Could not read:", image_id)
        continue

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Exclude nearly black background pixels
    mask = gray > 10

    if not np.any(mask):
        continue

    # Sharpness indicator
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpness = float(np.var(laplacian[mask]))

    # Brightness and contrast indicators
    brightness = float(np.mean(gray[mask]))
    contrast = float(np.std(gray[mask]))

    # Approximate non-black retinal coverage
    coverage = float(np.mean(mask) * 100)

    height, width = gray.shape

    results.append({
        "image_id": image_id,
        "true_grade": int(row["true_grade"]),
        "predicted_grade": int(row["predicted_grade"]),
        "correct": bool(row["correct"]),
        "width": width,
        "height": height,
        "sharpness": sharpness,
        "brightness": brightness,
        "contrast": contrast,
        "retinal_coverage_percent": coverage
    })

quality_df = pd.DataFrame(results)

quality_df.to_csv(OUTPUT_PATH, index=False)

print("\n===== IDRiD QUALITY ANALYSIS =====")
print("Images analyzed:", len(quality_df))
print("Images skipped:", len(predictions) - len(quality_df))
print("Saved:", OUTPUT_PATH)

print("\n===== OVERALL QUALITY: CORRECT VS INCORRECT =====")

metrics = [
    "sharpness",
    "brightness",
    "contrast",
    "retinal_coverage_percent"
]

summary = quality_df.groupby("correct")[metrics].agg(
    ["count", "median", "mean"]
)

print(summary.to_string())

grade_names = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "PDR"
}

print("\n===== SEVERE AND PDR QUALITY COMPARISON =====")

for grade in [3, 4]:
    subset = quality_df[quality_df["true_grade"] == grade]

    print(f"\n--- {grade_names[grade]} ---")
    print("Total images:", len(subset))
    print("Correct:", int(subset["correct"].sum()))
    print("Incorrect:", int((~subset["correct"]).sum()))

    comparison = subset.groupby("correct")[metrics].median()
    print(comparison.to_string())
