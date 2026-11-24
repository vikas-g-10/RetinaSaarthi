
"""Combined APTOS + IDRiD dataset using aspect-ratio-preserving transforms."""

from ml.classification.dataset_combined import (
    load_combined_splits,
    get_combined_class_weights,
)
from ml.classification.transforms_letterbox import (
    get_letterbox_transforms,
)


def load_combined_letterbox_splits():
    """Load the original combined splits with letterbox preprocessing."""

    train_dataset, val_dataset, test_dataset = load_combined_splits()

    train_dataset.transform = get_letterbox_transforms(training=True)
    val_dataset.transform = get_letterbox_transforms(training=False)
    test_dataset.transform = get_letterbox_transforms(training=False)

    print("\nLetterbox transforms applied to all partitions.")

    return train_dataset, val_dataset, test_dataset


__all__ = [
    "load_combined_letterbox_splits",
    "get_combined_class_weights",
]
