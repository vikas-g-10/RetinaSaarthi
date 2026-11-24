
"""Experiment 3: Baseline training strategy with LR scheduling."""

import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from ml.classification.dataset import load_splits, get_class_weights
from ml.classification.model import build_model


SEED = 42
BATCH_SIZE = 16
MAX_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5
NUM_WORKERS = 0

SCHEDULER_FACTOR = 0.5
SCHEDULER_PATIENCE = 2
MIN_LEARNING_RATE = 1e-7

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "classification"
    / "experiment3"
)

CHECKPOINT_PATH = OUTPUT_DIR / "best_model.pth"
HISTORY_PATH = OUTPUT_DIR / "training_history.csv"

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
):
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


def save_checkpoint(
    model,
    optimizer,
    epoch,
    best_f1,
    learning_rate,
):
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_macro_f1": best_f1,
            "seed": SEED,
            "class_names": CLASS_NAMES,
            "config": {
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "image_size": 224,
                "scheduler": "ReduceLROnPlateau",
                "maximum_epochs": MAX_EPOCHS,
            },
        },
        CHECKPOINT_PATH,
    )


def main():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading APTOS datasets...")

    train_dataset, val_dataset, _ = load_splits(seed=SEED)

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

    print("Loading pretrained EfficientNet-B0...")

    model = build_model(pretrained=True).to(device)

    class_weights = get_class_weights(train_dataset).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=SCHEDULER_FACTOR,
        patience=SCHEDULER_PATIENCE,
        min_lr=MIN_LEARNING_RATE,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    best_f1 = -1.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    history = []

    print("\nExperiment 3: Scheduled baseline training")
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Batch size:", BATCH_SIZE)
    print("Maximum epochs:", MAX_EPOCHS)
    print("Initial learning rate:", LEARNING_RATE)
    print("Weight decay:", WEIGHT_DECAY)

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

        # Scheduler monitors validation loss.
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch,
            "learning_rate": current_lr,
            "train_loss": train_loss,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
        })

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
            f"LR: {current_lr:.2e} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train F1: {train_f1:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_val_loss = val_loss
            epochs_without_improvement = 0

            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_f1,
                current_lr,
            )

            print("Saved improved macro-F1 checkpoint.")

        else:
            epochs_without_improvement += 1

            print(
                "No validation F1 improvement:",
                f"{epochs_without_improvement}/{PATIENCE}",
            )

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping triggered.")
            break

    with HISTORY_PATH.open("w", newline="") as file:
        fieldnames = [
            "epoch",
            "learning_rate",
            "train_loss",
            "train_macro_f1",
            "val_loss",
            "val_macro_f1",
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(history)

    print("\n" + "=" * 55)
    print("EXPERIMENT 3 COMPLETED")
    print("=" * 55)
    print(f"Best validation macro-F1: {best_f1:.4f}")
    print(f"Validation loss at best F1: {best_val_loss:.4f}")
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Training history:", HISTORY_PATH)
    print("Baseline and Experiment 2 checkpoints remain unchanged.")


if __name__ == "__main__":
    main()
