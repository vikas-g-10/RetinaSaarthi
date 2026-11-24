
"""Evaluate the letterbox combined model on the held-out APTOS test set."""

import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from ml.classification.dataset import CLASS_NAMES
from ml.classification.dataset_combined_letterbox import (
    load_combined_letterbox_splits,
)
from ml.classification.model import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "classification"
    / "combined_aptos_idrid_letterbox"
)

CHECKPOINT_PATH = OUTPUT_DIR / "best_model.pth"

BATCH_SIZE = 16
SEED = 42


def main():
    torch.manual_seed(SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

    _, _, test_dataset = load_combined_letterbox_splits()

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

            all_predictions.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.tolist())

    accuracy = accuracy_score(all_labels, all_predictions)

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
        "dataset": "APTOS 2019",
        "training_sources": ["APTOS", "IDRiD"],
        "preprocessing": "aspect_ratio_preserving_letterbox",
        "split": "held_out_aptos_test",
        "test_samples": len(test_dataset),
        "checkpoint_epoch": checkpoint["epoch"],
        "best_validation_macro_f1": checkpoint["best_val_macro_f1"],
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "class_names": CLASS_NAMES,
        "device": str(device),
    }

    with (OUTPUT_DIR / "test_metrics.json").open("w") as file:
        json.dump(metrics, file, indent=4)

    with (OUTPUT_DIR / "test_classification_report.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["class", "precision", "recall", "f1-score", "support"])

        for class_name in CLASS_NAMES:
            row = report[class_name]
            writer.writerow([
                class_name,
                row["precision"],
                row["recall"],
                row["f1-score"],
                row["support"],
            ])

    with (OUTPUT_DIR / "confusion_matrix.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["true/predicted"] + CLASS_NAMES)

        for index, row in enumerate(matrix):
            writer.writerow([CLASS_NAMES[index]] + row.tolist())

    with (OUTPUT_DIR / "test_predictions.csv").open(
        "w", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["image_id", "true_grade", "predicted_grade", "correct"])

        for index, (actual, predicted) in enumerate(
            zip(all_labels, all_predictions)
        ):
            image_id = test_dataset.dataframe.iloc[index]["id_code"]
            writer.writerow([image_id, actual, predicted, actual == predicted])

    print("\n========== LETTERBOX APTOS TEST RESULTS ==========")
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
    print("\nSaved results in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
