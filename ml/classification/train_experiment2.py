
"""Experiment 2: Two-stage EfficientNet-B0 fine-tuning."""

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
HEAD_EPOCHS = 3
FINETUNE_EPOCHS = 12
PATIENCE = 4
NUM_WORKERS = 0

HEAD_LR = 5e-4
BACKBONE_LR = 1e-5
CLASSIFIER_LR = 1e-4
LABEL_SMOOTHING = 0.05

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "classification" / "experiment2"
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


def set_backbone_trainable(model, trainable):
    """Freeze or unfreeze the EfficientNet feature extractor."""

    for parameter in model.features.parameters():
        parameter.requires_grad = trainable

    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
):
    """Run one training or validation epoch."""

    training = optimizer is not None

    model.train(training)

    # During classifier-only training, keep the frozen backbone
    # in evaluation mode so BatchNorm running statistics do not change.
    if training and not any(
        parameter.requires_grad
        for parameter in model.features.parameters()
    ):
        model.features.eval()

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


def save_checkpoint(model, epoch, best_f1):
    """Save the best validation checkpoint for Experiment 2."""

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "best_val_macro_f1": best_f1,
            "seed": SEED,
            "class_names": CLASS_NAMES,
            "experiment": "two_stage_finetuning",
        },
        CHECKPOINT_PATH,
    )


def main():
    set_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading APTOS training and validation datasets...")

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
        label_smoothing=LABEL_SMOOTHING,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda"),
    )

    history = []

    best_f1 = -1.0
    epochs_without_improvement = 0
    global_epoch = 0

    print("\nExperiment 2: Two-stage fine-tuning")
    print("Device:", device)
    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Batch size:", BATCH_SIZE)
    print("Label smoothing:", LABEL_SMOOTHING)

    # ---------------------------------------------------------
    # STAGE 1: Train classifier head while backbone is frozen.
    # ---------------------------------------------------------

    print("\n" + "=" * 55)
    print("STAGE 1: CLASSIFIER-ONLY TRAINING")
    print("=" * 55)

    set_backbone_trainable(model, False)

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(),
        lr=HEAD_LR,
        weight_decay=1e-3,
    )

    for epoch in range(1, HEAD_EPOCHS + 1):
        global_epoch += 1

        train_loss, train_f1 = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
        )

        val_loss, val_f1 = run_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        history.append({
            "epoch": global_epoch,
            "stage": "classifier_only",
            "train_loss": train_loss,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
        })

        print(
            f"Epoch {global_epoch:02d} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train F1: {train_f1:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(model, global_epoch, best_f1)
            print("Best checkpoint saved.")

    # ---------------------------------------------------------
    # STAGE 2: Unfreeze and fine-tune the complete network.
    # ---------------------------------------------------------

    print("\n" + "=" * 55)
    print("STAGE 2: FULL-NETWORK FINE-TUNING")
    print("=" * 55)

    set_backbone_trainable(model, True)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.features.parameters(),
                "lr": BACKBONE_LR,
            },
            {
                "params": model.classifier.parameters(),
                "lr": CLASSIFIER_LR,
            },
        ],
        weight_decay=1e-3,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
        min_lr=1e-7,
    )

    epochs_without_improvement = 0

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        global_epoch += 1

        train_loss, train_f1 = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
        )

        val_loss, val_f1 = run_epoch(
            model,
            val_loader,
            criterion,
            device,
        )

        scheduler.step(val_f1)

        history.append({
            "epoch": global_epoch,
            "stage": "full_finetuning",
            "train_loss": train_loss,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
        })

        current_backbone_lr = optimizer.param_groups[0]["lr"]
        current_classifier_lr = optimizer.param_groups[1]["lr"]

        print(
            f"Epoch {global_epoch:02d} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train F1: {train_f1:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val F1: {val_f1:.4f}"
        )

        print(
            f"Learning rates | "
            f"Backbone: {current_backbone_lr:.2e} | "
            f"Classifier: {current_classifier_lr:.2e}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            epochs_without_improvement = 0

            save_checkpoint(model, global_epoch, best_f1)
            print("Improved checkpoint saved.")

        else:
            epochs_without_improvement += 1
            print(
                "No validation F1 improvement: "
                f"{epochs_without_improvement}/{PATIENCE}"
            )

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping triggered.")
            break

    # ---------------------------------------------------------
    # SAVE TRAINING HISTORY.
    # ---------------------------------------------------------

    with HISTORY_PATH.open("w", newline="") as file:
        fieldnames = [
            "epoch",
            "stage",
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
    print("EXPERIMENT 2 COMPLETED")
    print("=" * 55)
    print(f"Best validation macro-F1: {best_f1:.4f}")
    print("Checkpoint:", CHECKPOINT_PATH)
    print("Training history:", HISTORY_PATH)
    print("Baseline checkpoint was not modified.")


if __name__ == "__main__":
    main()
