from pathlib import Path
import pandas as pd

base = Path("datasets/idrid/disease_grading/extracted")
grading_dir = base / "B. Disease Grading"
groundtruth_dir = grading_dir / "2. Groundtruths"
images_dir = grading_dir / "1. Original Images"

files = [
    (
        "Training",
        groundtruth_dir / "a. IDRiD_Disease Grading_Training Labels.csv",
        images_dir / "a. Training Set",
    ),
    (
        "Testing",
        groundtruth_dir / "b. IDRiD_Disease Grading_Testing Labels.csv",
        images_dir / "b. Testing Set",
    ),
]

for split, csv_path, image_dir in files:
    df = pd.read_csv(csv_path)

    print("\n" + "=" * 60)
    print(f"{split.upper()} SET")

    print("\nDR grade distribution:")
    print(df["Retinopathy grade"].value_counts().sort_index().to_string())

    print("\nMacular edema risk distribution:")
    print(
        df["Risk of macular edema "]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nDuplicate image names:", df["Image name"].duplicated().sum())

    labeled_names = set(df["Image name"].astype(str))
    image_names = {
        path.stem for path in image_dir.glob("*.jpg")
    }

    missing_images = sorted(labeled_names - image_names)
    unlabeled_images = sorted(image_names - labeled_names)

    print("Labels:", len(labeled_names))
    print("Image files:", len(image_names))
    print("Labels without images:", len(missing_images))
    print("Images without labels:", len(unlabeled_images))

    if missing_images:
        print("Missing image examples:", missing_images[:5])

    if unlabeled_images:
        print("Unlabeled image examples:", unlabeled_images[:5])
