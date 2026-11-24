"""Combined APTOS + IDRiD dataset using CLAHE preprocessing."""

from ml.classification.dataset_combined import (
    load_combined_splits,
    get_combined_class_weights,
)

from ml.classification.transforms_clahe import (
    get_clahe_transforms,
)


def load_combined_clahe_splits():
    """Load original combined splits with CLAHE preprocessing."""

    train_dataset, val_dataset, test_dataset = load_combined_splits()

    train_dataset.transform = get_clahe_transforms(
        training=True,
        clip_limit=1.0,
    )

    val_dataset.transform = get_clahe_transforms(
        training=False,
        clip_limit=1.0,
    )

    test_dataset.transform = get_clahe_transforms(
        training=False,
        clip_limit=1.0,
    )

    print("\nCLAHE transforms applied to all partitions.")
    print("CLAHE clip limit: 1.0")
    print("CLAHE tile grid: 8x8")

    return train_dataset, val_dataset, test_dataset


__all__ = [
    "load_combined_clahe_splits",
    "get_combined_class_weights",
]
