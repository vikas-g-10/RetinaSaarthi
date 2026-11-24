
"""APTOS dataset loading using a verified group-aware split manifest."""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CSV_PATH = PROJECT_ROOT / "datasets" / "aptos" / "train.csv"
IMAGE_DIR = PROJECT_ROOT / "datasets" / "aptos" / "train_images"

SPLIT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "classification"
    / "splits"
    / "aptos_grouped_seed42.csv"
)

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]


class APTOSDataset(Dataset):
    """Load APTOS fundus images and their DR grades."""

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = IMAGE_DIR / f"{row['id_code']}.png"

        image = Image.open(image_path).convert("RGB")
        label = int(row["diagnosis"])

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(training=False):
    """Return preprocessing transforms for pretrained EfficientNet-B0."""

    if training:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_splits(seed=42):
    """Load the permanent, verified group-aware APTOS split."""

    if seed != 42:
        raise ValueError(
            "Only the verified seed-42 manifest is available. "
            "Generate and verify another manifest before changing the seed."
        )

    if not CSV_PATH.exists():
        raise FileNotFoundError(f"APTOS CSV not found: {CSV_PATH}")

    if not SPLIT_PATH.exists():
        raise FileNotFoundError(
            f"Group-aware split manifest not found: {SPLIT_PATH}"
        )

    original_df = pd.read_csv(CSV_PATH)
    manifest = pd.read_csv(SPLIT_PATH)

    required_original = {"id_code", "diagnosis"}
    required_manifest = {
        "id_code",
        "diagnosis",
        "split",
        "duplicate_group",
    }

    if not required_original.issubset(original_df.columns):
        raise ValueError("APTOS CSV must contain id_code and diagnosis.")

    if not required_manifest.issubset(manifest.columns):
        raise ValueError("Split manifest is missing required columns.")

    if original_df[list(required_original)].isnull().any().any():
        raise ValueError("Original APTOS CSV contains missing values.")

    if manifest[list(required_manifest)].isnull().any().any():
        raise ValueError("Split manifest contains missing values.")

    if manifest["id_code"].duplicated().any():
        raise ValueError("Duplicate image IDs found in split manifest.")

    if not manifest["diagnosis"].isin(range(5)).all():
        raise ValueError("Unexpected DR grade found in manifest.")

    expected_splits = {"train", "validation", "test"}

    if set(manifest["split"].unique()) != expected_splits:
        raise ValueError("Manifest must contain train, validation and test.")

    # Verify that manifest IDs and labels match the original CSV.
    original_labels = original_df.set_index("id_code")["diagnosis"]

    if not manifest["id_code"].isin(original_labels.index).all():
        raise ValueError("Manifest contains IDs absent from original CSV.")

    expected_labels = manifest["id_code"].map(original_labels)

    if not (expected_labels.to_numpy() == manifest["diagnosis"].to_numpy()).all():
        raise ValueError("Manifest labels do not match original CSV.")

    # Verify that no exact-duplicate group crosses partitions.
    group_counts = manifest.groupby("duplicate_group")["split"].nunique()

    if (group_counts > 1).any():
        raise ValueError("Duplicate leakage detected in split manifest.")

    # Verify every class exists in every partition.
    class_counts = pd.crosstab(
        manifest["split"],
        manifest["diagnosis"],
    )

    if (
        not expected_splits.issubset(set(class_counts.index))
        or len(class_counts.columns) != 5
        or (class_counts == 0).any().any()
    ):
        raise ValueError("Every split must contain all five DR grades.")

    # Verify all referenced image files exist.
    missing_images = [
        image_id
        for image_id in manifest["id_code"]
        if not (IMAGE_DIR / f"{image_id}.png").is_file()
    ]

    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} manifest images are missing. "
            f"Examples: {missing_images[:5]}"
        )

    train_df = manifest[manifest["split"] == "train"].copy()
    val_df = manifest[manifest["split"] == "validation"].copy()
    test_df = manifest[manifest["split"] == "test"].copy()

    print("Loaded verified group-aware APTOS manifest.")
    print(
        f"Train: {len(train_df)} | "
        f"Validation: {len(val_df)} | "
        f"Test: {len(test_df)}"
    )

    return (
        APTOSDataset(train_df, get_transforms(training=True)),
        APTOSDataset(val_df, get_transforms()),
        APTOSDataset(test_df, get_transforms()),
    )


def get_class_weights(train_dataset):
    """Calculate inverse-frequency class weights from training data only."""

    labels = train_dataset.dataframe["diagnosis"]
    counts = labels.value_counts().sort_index()

    if len(counts) != 5 or (counts == 0).any():
        raise ValueError("Every DR grade must appear in training data.")

    weights = len(labels) / (5.0 * counts)

    return torch.tensor(weights.values, dtype=torch.float32)
