
"""Combined APTOS + IDRiD training dataset with fixed APTOS evaluation."""

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from ml.classification.dataset import (
    PROJECT_ROOT,
    SPLIT_PATH,
    IMAGE_DIR,
    get_transforms,
)


COMBINED_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "classification"
    / "splits"
    / "combined_train_aptos_idrid.csv"
)


class CombinedFundusDataset(Dataset):
    """Load fundus images from multiple datasets using manifest paths."""

    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        image_path = PROJECT_ROOT / row["image_path"]

        with Image.open(image_path) as image_file:
            image = image_file.convert("RGB")

        label = int(row["diagnosis"])

        if self.transform:
            image = self.transform(image)

        return image, label


def load_combined_splits():
    """Load combined training data and fixed APTOS validation/test data."""

    if not COMBINED_MANIFEST.is_file():
        raise FileNotFoundError(
            f"Combined manifest not found: {COMBINED_MANIFEST}"
        )

    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(
            f"APTOS split manifest not found: {SPLIT_PATH}"
        )

    combined = pd.read_csv(COMBINED_MANIFEST)
    aptos_splits = pd.read_csv(SPLIT_PATH)

    required_combined = {
        "source",
        "id_code",
        "diagnosis",
        "image_path",
        "duplicate_group",
    }

    if not required_combined.issubset(combined.columns):
        raise ValueError("Combined manifest is missing required columns.")

    if combined[list(required_combined)].isnull().any().any():
        raise ValueError("Combined manifest contains missing values.")

    if not combined["diagnosis"].isin(range(5)).all():
        raise ValueError("Unexpected DR grade in combined manifest.")

    if combined["image_path"].duplicated().any():
        raise ValueError("Duplicate image paths in combined manifest.")

    if combined.duplicated(["source", "id_code"]).any():
        raise ValueError("Duplicate source/image IDs in combined manifest.")

    if set(combined["source"].unique()) != {"APTOS", "IDRiD"}:
        raise ValueError("Unexpected dataset source in combined manifest.")

    if combined.groupby("duplicate_group")["source"].nunique().gt(1).any():
        raise ValueError("Duplicate group shared across dataset sources.")

    missing = [
        path
        for path in combined["image_path"]
        if not (PROJECT_ROOT / path).is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} training images are missing: {missing[:5]}"
        )

    # Preserve the exact original APTOS evaluation partitions.
    val_df = aptos_splits[
        aptos_splits["split"] == "validation"
    ].copy()

    test_df = aptos_splits[
        aptos_splits["split"] == "test"
    ].copy()

    for frame, name in [(val_df, "validation"), (test_df, "test")]:
        if frame.empty:
            raise ValueError(f"APTOS {name} partition is empty.")

        if not frame["diagnosis"].isin(range(5)).all():
            raise ValueError(f"Invalid labels in APTOS {name} partition.")

        missing_ids = [
            image_id
            for image_id in frame["id_code"]
            if not (IMAGE_DIR / f"{image_id}.png").is_file()
        ]

        if missing_ids:
            raise FileNotFoundError(
                f"Missing APTOS {name} images: {missing_ids[:5]}"
            )

    print("Loaded combined APTOS + IDRiD dataset.")
    print("Training source counts:")
    print(combined["source"].value_counts().to_string())

    print("\nTraining grade counts:")
    print(combined["diagnosis"].value_counts().sort_index().to_string())

    print(
        f"\nTrain: {len(combined)} | "
        f"Validation: {len(val_df)} | "
        f"Test: {len(test_df)}"
    )

    return (
        CombinedFundusDataset(
            combined,
            get_transforms(training=True),
        ),
        CombinedFundusDataset(
            val_df.assign(
                image_path=val_df["id_code"].map(
                    lambda x: (
                        IMAGE_DIR / f"{x}.png"
                    ).relative_to(PROJECT_ROOT).as_posix()
                )
            ),
            get_transforms(),
        ),
        CombinedFundusDataset(
            test_df.assign(
                image_path=test_df["id_code"].map(
                    lambda x: (
                        IMAGE_DIR / f"{x}.png"
                    ).relative_to(PROJECT_ROOT).as_posix()
                )
            ),
            get_transforms(),
        ),
    )


def get_combined_class_weights(train_dataset):
    """Calculate inverse-frequency weights from combined training labels."""

    labels = train_dataset.dataframe["diagnosis"]
    counts = labels.value_counts().reindex(range(5), fill_value=0)

    if (counts == 0).any():
        raise ValueError("Every DR grade must appear in training data.")

    weights = len(labels) / (5.0 * counts)

    return torch.tensor(weights.to_numpy(), dtype=torch.float32)
