
"""Analyze validation predictions for the combined ordinal DR model."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from ml.classification.dataset import CLASS_NAMES
from ml.classification.dataset_combined import load_combined_splits
from ml.classification.model_ordinal import (
    build_ordinal_model,
    decode_ordinal_logits,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "classification"
    / "combined_aptos_idrid_ordinal"
)

CHECKPOINT_PATH = MODEL_DIR / "best_model.pth"
OUTPUT_DIR = MODEL_DIR / "validation_analysis"

BATCH_SIZE = 16


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(CHECKPOINT_PATH)

    _, val_dataset, _ = load_combined_splits()

    loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = build_ordinal_model(pretrained=False).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_labels = []
    all_logits = []

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                logits = model(images)

            all_logits.extend(logits.float().cpu().numpy())
            all_labels.extend(labels.numpy())

    logits = np.asarray(all_logits)
    actual = np.asarray(all_labels)

    # Four cumulative probabilities: P(grade > 0), ..., P(grade > 3).
    ordinal_probabilities = 1.0 / (1.0 + np.exp(-logits))

    predicted = (
        ordinal_probabilities >= 0.5
    ).sum(axis=1)

    # Mean confidence of the four binary decisions.
    binary_confidence = np.maximum(
        ordinal_probabilities,
        1.0 - ordinal_probabilities,
    ).mean(axis=1)

    image_ids = val_dataset.dataframe["id_code"].tolist()

    results = pd.DataFrame({
        "image_id": image_ids,
        "true_grade": actual,
        "predicted_grade": predicted,
        "correct": actual == predicted,
        "confidence": binary_confidence,
        "grade_error": predicted - actual,
        "absolute_grade_error": np.abs(predicted - actual),
    })

    for index in range(4):
        results[f"prob_grade_gt_{index}"] = (
            ordinal_probabilities[:, index]
        )

    accuracy = accuracy_score(actual, predicted)

    macro_f1 = f1_score(
        actual,
        predicted,
        labels=list(range(5)),
        average="macro",
        zero_division=0,
    )

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=list(range(5)),
    )

    report = classification_report(
        actual,
        predicted,
        labels=list(range(5)),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    summary = {
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "best_validation_macro_f1": float(
            checkpoint["best_val_macro_f1"]
        ),
        "validation_samples": len(actual),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "mean_binary_confidence": float(binary_confidence.mean()),
        "mean_absolute_grade_error": float(
            results["absolute_grade_error"].mean()
        ),
        "within_one_grade_accuracy": float(
            (results["absolute_grade_error"] <= 1).mean()
        ),
        "large_grade_error_count": int(
            (results["absolute_grade_error"] >= 2).sum()
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results.to_csv(
        OUTPUT_DIR / "validation_predictions.csv",
        index=False,
    )

    with (OUTPUT_DIR / "validation_metrics.json").open("w") as file:
        json.dump(summary, file, indent=4)

    pd.DataFrame(
        matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    pd.DataFrame(report).transpose().to_csv(
        OUTPUT_DIR / "classification_report.csv"
    )

    print("\n========== ORDINAL VALIDATION ANALYSIS ==========")
    print("Device:", device)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Validation samples:", len(actual))
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Mean binary confidence: {binary_confidence.mean():.4f}")
    print(
        "Mean absolute grade error:",
        f"{summary['mean_absolute_grade_error']:.4f}",
    )
    print(
        "Within-one-grade accuracy:",
        f"{summary['within_one_grade_accuracy']:.4f}",
    )
    print(
        "Errors of 2 or more grades:",
        summary["large_grade_error_count"],
    )

    print("\nPer-class results:")
    print(
        classification_report(
            actual,
            predicted,
            labels=list(range(5)),
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(matrix)

    print("\nSaved analysis to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
