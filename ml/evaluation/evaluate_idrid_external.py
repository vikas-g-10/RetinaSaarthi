
"""Evaluate the combined APTOS + IDRiD model on the IDRiD external test set."""

import csv
import json
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from ml.classification.dataset import CLASS_NAMES, get_transforms
from ml.classification.model import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "classification"
    / "combined_aptos_idrid_baseline"
    / "external_idrid"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "classification"
    / "combined_aptos_idrid_baseline"
    / "best_model.pth"
)

IDRID_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "idrid"
    / "disease_grading"
    / "extracted"
    / "B. Disease Grading"
)

CSV_PATH = (
    IDRID_ROOT
    / "2. Groundtruths"
    / "b. IDRiD_Disease Grading_Testing Labels.csv"
)

IMAGE_DIR = (
    IDRID_ROOT
    / "1. Original Images"
    / "b. Testing Set"
)

BATCH_SIZE = 16
SEED = 42


class IDRiDTestDataset(Dataset):
    """Load IDRiD testing images with evaluation preprocessing."""

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = Path(row["image_path"])

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        label = int(row["Retinopathy grade"])

        if self.transform:
            image = self.transform(image)

        return image, label


def load_idrid_test():
    """Load and validate the IDRiD external testing dataset."""

    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"IDRiD test CSV not found: {CSV_PATH}")

    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"IDRiD image directory not found: {IMAGE_DIR}")

    df = pd.read_csv(CSV_PATH)

    required_columns = {"Image name", "Retinopathy grade"}

    if not required_columns.issubset(df.columns):
        raise ValueError("IDRiD testing CSV is missing required columns.")

    df["Image name"] = df["Image name"].astype(str).str.strip()

    if df["Image name"].duplicated().any():
        raise ValueError("Duplicate image IDs in IDRiD testing CSV.")

    if df["Retinopathy grade"].isna().any():
        raise ValueError("Missing DR labels in IDRiD testing CSV.")

    df["Retinopathy grade"] = df["Retinopathy grade"].astype(int)

    if not df["Retinopathy grade"].isin(range(5)).all():
        raise ValueError("Invalid DR grades in IDRiD testing CSV.")

    df["image_path"] = df["Image name"].apply(
        lambda name: IMAGE_DIR / f"{name}.jpg"
    )

    missing = [
        str(path)
        for path in df["image_path"]
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} IDRiD test images missing: {missing[:5]}"
        )

    dataset = IDRiDTestDataset(
        df,
        transform=get_transforms(training=False),
    )

    print("Loaded IDRiD external test dataset.")
    print("Grade distribution:")
    print(
        df["Retinopathy grade"]
        .value_counts()
        .sort_index()
        .to_string()
    )
    print("\nTotal test images:", len(dataset))

    return dataset


def main():
    torch.manual_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}"
        )

    test_dataset = load_idrid_test()

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(pretrained=False).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.inference_mode():
        for images, labels in test_loader:
            images = images.to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                outputs = model(images)

            predictions = outputs.argmax(dim=1).cpu().tolist()

            all_predictions.extend(predictions)
            all_labels.extend(labels.tolist())

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=list(range(5)),
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        all_labels,
        all_predictions,
        labels=list(range(5)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=list(range(5)),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics = {
        "dataset": "IDRiD",
        "evaluation_type": "external_test",
        "training_sources": ["APTOS", "IDRiD"],
        "test_samples": len(test_dataset),
        "checkpoint_epoch": checkpoint["epoch"],
        "best_validation_macro_f1": checkpoint["best_val_macro_f1"],
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "class_names": CLASS_NAMES,
        "device": str(device),
    }

    with (OUTPUT_DIR / "external_metrics.json").open("w") as file:
        json.dump(metrics, file, indent=4)

    with (OUTPUT_DIR / "external_classification_report.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "class",
            "precision",
            "recall",
            "f1-score",
            "support",
        ])

        for class_name in CLASS_NAMES:
            row = report[class_name]
            writer.writerow([
                class_name,
                row["precision"],
                row["recall"],
                row["f1-score"],
                row["support"],
            ])

    with (OUTPUT_DIR / "external_confusion_matrix.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["true/predicted"] + CLASS_NAMES)

        for index, row in enumerate(matrix):
            writer.writerow([
                CLASS_NAMES[index],
                *row.tolist(),
            ])

    with (OUTPUT_DIR / "external_predictions.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "image_id",
            "true_grade",
            "predicted_grade",
            "correct",
        ])

        for index, (actual, predicted) in enumerate(
            zip(all_labels, all_predictions)
        ):
            image_id = test_dataset.dataframe.iloc[index]["Image name"]

            writer.writerow([
                image_id,
                actual,
                predicted,
                actual == predicted,
            ])

    print("\n========== IDRiD EXTERNAL TEST RESULTS ==========")
    print("Device:", device)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Test samples:", len(test_dataset))
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test macro-F1: {macro_f1:.4f}")

    print("\nPer-class results:")
    print(classification_report(
        all_labels,
        all_predictions,
        labels=list(range(5)),
        target_names=CLASS_NAMES,
        zero_division=0,
    ))

    print("Confusion matrix:")
    print(matrix)

    print("\nSaved external evaluation files in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
