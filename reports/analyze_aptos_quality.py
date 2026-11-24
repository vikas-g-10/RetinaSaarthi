import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

csv_path = Path("datasets/aptos/train.csv")
image_dir = Path("datasets/aptos/train_images")
output_path = Path("reports/aptos_quality_metrics.csv")

df = pd.read_csv(csv_path)
results = []

print("Analyzing image quality for", len(df), "images...")

for _, row in tqdm(df.iterrows(), total=len(df)):
    image_path = image_dir / f"{row['id_code']}.png"

    image = cv2.imread(str(image_path))

    if image is None:
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
        "id_code": row["id_code"],
        "diagnosis": int(row["diagnosis"]),
        "width": width,
        "height": height,
        "sharpness": sharpness,
        "brightness": brightness,
        "contrast": contrast,
        "retinal_coverage_percent": coverage
    })

result_df = pd.DataFrame(results)
result_df.to_csv(output_path, index=False)

print("\nAnalysis completed.")
print("Images analyzed:", len(result_df))
print("Images skipped:", len(df) - len(result_df))
print("Saved to:", output_path)
