
from pathlib import Path
import random
import hashlib

import cv2
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = ROOT / "reports/quality_metrics_standardised.csv"
OUT = ROOT / "reports/quality_review_v3"

SEED = 26038
RANDOM_PER_DATASET = 30
TARGETED_PER_DATASET = 20

DATASETS = ["APTOS", "Messidor-2", "IDRiD-segmentation"]

METRICS = [
    "sharpness",
    "illumination_mean",
    "illumination_std",
    "illumination_quadrant_ratio",
    "contrast_p5_p95",
    "field_coverage_percent",
]


def add_reason(reasons, index, reason):
    reasons.setdefault(index, set()).add(reason)


def main():
    if not SOURCE_CSV.is_file():
        raise FileNotFoundError(SOURCE_CSV)

    if OUT.exists():
        raise FileExistsError(
            f"{OUT} already exists. Do not overwrite an existing review."
        )

    df = pd.read_csv(SOURCE_CSV)

    required = {
        "dataset", "file_name", "grade", "width", "height",
        *METRICS, "field_touches_edge"
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {sorted(missing)}")

    rng = random.Random(SEED)
    selections = {}

    # 1. Select a stratified random sample per dataset.
    for dataset_number, dataset in enumerate(DATASETS):
        group = df[df.dataset == dataset]

        if len(group) < RANDOM_PER_DATASET:
            raise ValueError(f"Not enough images in {dataset}")

        selected = group.sample(
            n=RANDOM_PER_DATASET,
            random_state=SEED + dataset_number,
        )

        for index in selected.index:
            selections[index] = {"random_sample"}

    # 2. Select targeted samples without overlapping random samples.
    for dataset in DATASETS:
        group = df[df.dataset == dataset]
        available = group.loc[
            ~group.index.isin(selections.keys())
        ]

        reasons = {}

        # Two lowest and two highest examples per metric.
        for metric in METRICS:
            valid = available.dropna(subset=[metric])

            for index in valid.nsmallest(2, metric).index:
                add_reason(reasons, index, f"{metric}_low")

            for index in valid.nlargest(2, metric).index:
                add_reason(reasons, index, f"{metric}_high")

        # Images around the quadrant-ratio transition region.
        valid = available.dropna(
            subset=["illumination_quadrant_ratio"]
        )

        transition = (
            valid.assign(
                distance=(
                    valid.illumination_quadrant_ratio - 2.4
                ).abs()
            )
            .nsmallest(5, "distance")
        )

        for index in transition.index:
            add_reason(
                reasons, index, "quadrant_ratio_transition"
            )

        # Include a small number of edge-contact examples.
        edge = available[
            available.field_touches_edge.astype(str).str.lower()
            .isin(["true", "1"])
        ]

        if len(edge):
            edge_indices = rng.sample(
                list(edge.index),
                min(3, len(edge)),
            )
            for index in edge_indices:
                add_reason(reasons, index, "field_edge_contact")

        # Prefer images selected by multiple criteria.
        candidates = list(reasons)
        rng.shuffle(candidates)
        candidates.sort(
            key=lambda index: len(reasons[index]),
            reverse=True,
        )

        chosen = candidates[:TARGETED_PER_DATASET]

        # Fill any remaining places with reproducible random selections.
        remaining = [
            index for index in available.index
            if index not in chosen
        ]
        rng.shuffle(remaining)

        chosen += remaining[
            :TARGETED_PER_DATASET - len(chosen)
        ]

        for index in chosen:
            selections[index] = reasons.get(
                index, {"targeted_fill"}
            )

    # 3. Shuffle the combined sample before assigning blind IDs.
    selected_indices = list(selections)
    rng.shuffle(selected_indices)

    OUT.mkdir(parents=True)
    image_dir = OUT / "review_images"
    image_dir.mkdir()

    manifest = []
    reviewer_rows = []
    failures = []

    used_ids = set()

    for index in selected_indices:
        row = df.loc[index]
        source = ROOT / str(row.file_name)

        if not source.is_file():
            failures.append({
                "source": str(row.file_name),
                "reason": "Source image not found",
            })
            continue

        image = cv2.imread(str(source), cv2.IMREAD_COLOR)

        if image is None:
            failures.append({
                "source": str(row.file_name),
                "reason": "OpenCV could not decode image",
            })
            continue

        # Re-encode all images as PNG:
        # same decoded visual pixels, normalized extension,
        # and no original file metadata copied.
        blind_id = None

        while blind_id is None or blind_id in used_ids:
            blind_id = f"R-{rng.getrandbits(32):08X}"

        used_ids.add(blind_id)

        image_name = f"{blind_id}.png"
        image_path = image_dir / image_name

        if not cv2.imwrite(str(image_path), image):
            failures.append({
                "source": str(row.file_name),
                "reason": "Could not write review PNG",
            })
            continue

        reasons = sorted(selections[index])

        record = row.to_dict()
        record.update({
            "blind_id": blind_id,
            "review_image": f"review_images/{image_name}",
            "sampling_group": (
                "random" if "random_sample" in reasons
                else "targeted"
            ),
            "sampling_reasons": ";".join(reasons),
            "source_sha256": hashlib.sha256(
                source.read_bytes()
            ).hexdigest(),
        })

        manifest.append(record)

        reviewer_rows.append({
            "blind_id": blind_id,
            "image_file": f"review_images/{image_name}",
            "image_quality": "",
            "field_completeness": "",
            "field_size_framing": "",
            "blur_at_native_resolution": "",
            "usability_decision": "",
            "defect_tags": "",
            "reviewer_id": "",
            "review_date": "",
            "notes": "",
        })

    manifest_df = pd.DataFrame(manifest)
    reviewer_df = pd.DataFrame(reviewer_rows)

    manifest_df.to_csv(
        OUT / "sampling_manifest.csv", index=False
    )
    reviewer_df.to_csv(
        OUT / "review_labels.csv", index=False
    )

    pd.DataFrame(failures).to_csv(
        OUT / "image_copy_failures.csv", index=False
    )

    codebook = """# RetinaSaarthi — Blinded Image Quality Review

## Instructions

Review the original-resolution appearance of each displayed image.
Do not use DR grade, model predictions, or quality metric values
to make the human assessment.

## Image quality
- good: visually suitable for screening
- borderline: some degradation, usability uncertain
- poor: substantial image-quality defect
- uncertain: cannot assess confidently

## Field completeness
- complete: retinal field appears adequately captured
- partially_clipped: part of the retinal field appears cut off
- obstructed: a region is obscured
- uncertain: cannot determine

## Field size / framing
- small: relatively small retinal field in the frame
- typical: framing appears reasonable
- large: retinal field occupies most of the frame
- uncertain: cannot determine

Field size is not the same as field completeness.

## Blur at native resolution
- yes: visible blur
- no: no obvious blur
- uncertain: cannot determine

Inspect vessels and fine structures at native resolution.

## Usability decision
- usable
- recapture
- manual_review

A mask-detection failure alone must not automatically mean recapture.

## Defect tags

Use semicolon-separated tags where applicable:
dark; overexposed; glare; colour_cast; haze; blur;
uneven_illumination; clipped; obstructed; lesion_heavy; other

Lesion-heavy is a visual observation, not a diagnosis.

## Notes

Describe visible reasons for borderline, recapture, or manual review.
Do not infer DR grade from this review.
"""

    (OUT / "review_codebook.md").write_text(
        codebook, encoding="utf-8"
    )

    print("\nREVIEW PACKAGE CREATED")
    print("Output:", OUT.relative_to(ROOT))
    print("Images copied:", len(manifest_df))
    print("\nSampling groups:")
    print(
        manifest_df.groupby(
            ["dataset", "sampling_group"]
        ).size().to_string()
    )
    print("\nCopy failures:", len(failures))
    print("Reviewer CSV:", OUT / "review_labels.csv")
    print("Private manifest:", OUT / "sampling_manifest.csv")


if __name__ == "__main__":
    main()
