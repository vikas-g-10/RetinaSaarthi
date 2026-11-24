
"""Train EfficientNet-B0 with class-weighted focal loss."""

import csv
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from ml.classification.dataset_combined import (
    load_combined_splits,
    get_combined_class_weights,
)
from ml.classification.model import build_model
from ml.classification.losses import ClassWeightedFocalLoss


SEED = 42
BATCH_SIZE = 16
MAX_EPOCHS = 10
LEARNING_RATE = 1e-4
PATIENCE = 3
NUM_WORKERS = 0
FOCAL_GAMMA = 2.0

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "classification"
    / "combined_aptos_idrid_focal"
)

CHECKPOINT_PATH = OUTPUT_DIR / "best_model.pth"
HISTORY_PATH = OUTPUT_DIR / "training_history.csv"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    all_labels = []
    all_predictions = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.detach().cpu().tolist())
        all_predictions.extend(predictions.detach().cpu().tolist())

    average_loss = total_loss / len(loader.dataset)

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        labels=[0, 1, 2, 3, 4],
        average="macro",
        zero_division=0,
    )

    return average_loss, macro_f1


def main():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    train_dataset, val_dataset, _ = load_combined_splits()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(pretrained=True).to(device)

    class_weights = get_combined_class_weights(
        train_dataset
    ).to(device)

    criterion = ClassWeightedFocalLoss(
        class_weights=class_weights,
        gamma=FOCAL_GAMMA,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    best_f1 = -1.0
    epochs_without_improvement = 0
    history = []

    print("\nTraining images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Batch size:", BATCH_SIZE)
    print("Maximum epochs:", MAX_EPOCHS)
    print("Learning rate:", LEARNING_RATE)
    print("Focal gamma:", FOCAL_GAMMA)
    print("Class weights:", class_weights.tolist())

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_f1 = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
        )

        val_loss, val_f1 = run_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
        }

        history.append(row)

        print(
            f"Epoch {epoch}/{MAX_EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train macro-F1: {train_f1:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val macro-F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_without_improvement = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_macro_f1": best_f1,
                    "seed": SEED,
                    "class_names": [
                        "No DR",
                        "Mild",
                        "Moderate",
                        "Severe",
                        "Proliferative DR",
                    ],
                    "config": {
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "image_size": 224,
                        "training_sources": ["APTOS", "IDRiD"],
                        "loss": "class_weighted_focal",
                        "focal_gamma": FOCAL_GAMMA,
                        "class_weights": class_weights.tolist(),
                    },
                },
                CHECKPOINT_PATH,
            )

            print("Saved best checkpoint:", CHECKPOINT_PATH)

        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping: validation macro-F1 did not improve.")
            break

    with HISTORY_PATH.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print("\nFocal-loss training finished.")
    print(f"Best validation macro-F1: {best_f1:.4f}")
    print("Checkpoint:", CHECKPOINT_PATH)
    print("History:", HISTORY_PATH)


if __name__ == "__main__":
    main()
