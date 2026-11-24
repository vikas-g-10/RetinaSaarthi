
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]

MANIFEST = ROOT / "outputs/classification/splits/combined_train_aptos_idrid.csv"

IDRID_TEST = (
    ROOT / "outputs/classification/combined_aptos_idrid_baseline/"
    "external_idrid/external_predictions.csv"
)

IDRID_TEST_DIR = (
    ROOT / "datasets/idrid/disease_grading/extracted/B. Disease Grading/"
    "1. Original Images/b. Testing Set"
)

OUTPUT = ROOT / "reports/dataset_characteristics.csv"


def analyze_image(path, source, image_id, grade):
    image = cv2.imread(str(path))

    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask = gray > 10

    if not np.any(mask):
        return None

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    height, width = gray.shape

    pixels = rgb[mask]

    return {
        "source": source,
        "image_id": image_id,
        "grade": int(grade),
        "width": width,
        "height": height,
        "aspect_ratio": width / height,
        "sharpness": float(np.var(laplacian[mask])),
        "brightness": float(np.mean(gray[mask])),
        "contrast": float(np.std(gray[mask])),
        "coverage_percent": float(np.mean(mask) * 100),
        "red_mean": float(np.mean(pixels[:, 0])),
        "green_mean": float(np.mean(pixels[:, 1])),
        "blue_mean": float(np.mean(pixels[:, 2])),
        "red_std": float(np.std(pixels[:, 0])),
        "green_std": float(np.std(pixels[:, 1])),
        "blue_std": float(np.std(pixels[:, 2]))
    }


def main():
    manifest = pd.read_csv(MANIFEST)
    test = pd.read_csv(IDRID_TEST)

    records = []

    print("===== ANALYZING TRAINING IMAGES =====")

    for row in tqdm(manifest.itertuples(index=False), total=len(manifest)):
        path = ROOT / row.image_path

        result = analyze_image(
            path, row.source, row.id_code, row.diagnosis
        )

        if result:
            records.append(result)

    print("\n===== ANALYZING IDRiD TEST IMAGES =====")

    for row in tqdm(test.itertuples(index=False), total=len(test)):
        path = IDRID_TEST_DIR / f"{row.image_id}.jpg"

        result = analyze_image(
            path, "IDRiD_TEST", row.image_id, row.true_grade
        )

        if result:
            records.append(result)

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT, index=False)

    print("\n===== IMAGE COUNTS =====")
    print(df["source"].value_counts().to_string())

    metrics = [
        "width", "height", "aspect_ratio",
        "sharpness", "brightness", "contrast",
        "coverage_percent",
        "red_mean", "green_mean", "blue_mean",
        "red_std", "green_std", "blue_std"
    ]

    print("\n===== MEDIAN CHARACTERISTICS BY DATASET =====")
    print(df.groupby("source")[metrics].median().round(2).to_string())

    print("\n===== MEAN CHARACTERISTICS BY DATASET =====")
    print(df.groupby("source")[metrics].mean().round(2).to_string())

    print("\n===== RESOLUTION DISTRIBUTION =====")
    for source, subset in df.groupby("source"):
        print(f"\n{source}:")
        print(
            subset.groupby(["width", "height"]).size()
            .sort_values(ascending=False).head(8).to_string()
        )

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
