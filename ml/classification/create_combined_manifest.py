
"""Create a verified combined APTOS + IDRiD training manifest."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

APTOS_CSV = ROOT / "datasets/aptos/train.csv"
APTOS_SPLIT = (
    ROOT / "outputs/classification/splits/aptos_grouped_seed42.csv"
)
APTOS_IMAGES = ROOT / "datasets/aptos/train_images"

IDRID_BASE = ROOT / "datasets/idrid/disease_grading/extracted/B. Disease Grading"
IDRID_CSV = (
    IDRID_BASE
    / "2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv"
)
IDRID_IMAGES = IDRID_BASE / "1. Original Images/a. Training Set"

OUTPUT = (
    ROOT
    / "outputs/classification/splits/combined_train_aptos_idrid.csv"
)

# Exclude the train/test duplicate, both conflicting-label pairs,
# and one redundant copy from each matching-label pair.
EXCLUDED_IDRID = {
    "IDRiD_118",
    "IDRiD_021", "IDRiD_391",
    "IDRiD_028", "IDRiD_327",
    "IDRiD_333",
    "IDRiD_375",
    "IDRiD_279",
}


def main():
    aptos = pd.read_csv(APTOS_CSV)
    aptos_split = pd.read_csv(APTOS_SPLIT)
    idrid = pd.read_csv(IDRID_CSV)

    # Use only the verified APTOS training partition.
    aptos_train = aptos_split[
        aptos_split["split"] == "train"
    ].copy()

    aptos_train["source"] = "APTOS"
    aptos_train["image_path"] = aptos_train["id_code"].map(
        lambda x: (APTOS_IMAGES / f"{x}.png").relative_to(ROOT).as_posix()
    )
    aptos_train["duplicate_group"] = (
        "APTOS:" + aptos_train["duplicate_group"].astype(str)
    )

    # Use only the cleaned IDRiD training records.
    idrid = idrid[
        ~idrid["Image name"].isin(EXCLUDED_IDRID)
    ].copy()

    idrid_train = pd.DataFrame({
        "id_code": idrid["Image name"],
        "diagnosis": idrid["Retinopathy grade"].astype(int),
        "source": "IDRiD",
    })

    idrid_train["image_path"] = idrid_train["id_code"].map(
        lambda x: (
            IDRID_IMAGES / f"{x}.jpg"
        ).relative_to(ROOT).as_posix()
    )

    # Each retained IDRiD image is its own duplicate group.
    idrid_train["duplicate_group"] = (
        "IDRiD:" + idrid_train["id_code"]
    )

    columns = [
        "source",
        "id_code",
        "diagnosis",
        "image_path",
        "duplicate_group",
    ]

    combined = pd.concat(
        [
            aptos_train[columns],
            idrid_train[columns],
        ],
        ignore_index=True,
    )

    # Verify the manifest before saving.
    assert len(aptos_train) == 2519
    assert len(idrid_train) == 405
    assert len(combined) == 2924
    assert not combined["diagnosis"].isna().any()
    assert combined["diagnosis"].isin(range(5)).all()
    assert not combined["image_path"].duplicated().any()
    assert not combined.duplicated(["source", "id_code"]).any()

    missing = [
        p for p in combined["image_path"]
        if not (ROOT / p).is_file()
    ]
    assert not missing, f"Missing images: {missing[:10]}"

    print("===== COMBINED TRAINING MANIFEST =====")
    print("APTOS records:", len(aptos_train))
    print("IDRiD records:", len(idrid_train))
    print("Total records:", len(combined))

    print("\nSource distribution:")
    print(combined["source"].value_counts())

    print("\nCombined DR grade distribution:")
    print(combined["diagnosis"].value_counts().sort_index())

    print("\nDuplicate image IDs:", combined.duplicated(
        ["source", "id_code"]
    ).sum())
    print("Missing image files:", len(missing))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT, index=False)

    print("\nManifest saved:", OUTPUT)


if __name__ == "__main__":
    main()
