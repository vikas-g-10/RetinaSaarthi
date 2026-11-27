"""Evaluate prespecified IDRiD candidate overlap; see reports/objective3_protocol.md."""
import csv
import argparse

import numpy as np
from PIL import Image

from ml.explainability.generate_gradcam import (
    DEVICE, INFERENCE_TRANSFORM, LESION_CATEGORIES, MASK_ROOT, ROOT,
    generate_gradcam, load_model,
)

SEG = ROOT / 'datasets/idrid/segmentation/extracted/A. Segmentation/1. Original Images/a. Training Set'
GRADE = ROOT / 'datasets/idrid/disease_grading/extracted/B. Disease Grading/1. Original Images/b. Testing Set'
OUT = ROOT / 'outputs/explainability/idrid_overlap'
PAIRS = [('09', '091'), ('22', '088'), ('31', '089')]


def overlap(activation, lesion):
    """Return IoU, Dice and fraction of recorded lesion pixels covered."""
    a, l = int(activation.sum()), int(lesion.sum())
    if not l:
        raise ValueError('Empty lesion union is ineligible')
    intersection = int(np.count_nonzero(activation & lesion))
    return intersection / (a + l - intersection), 2 * intersection / (a + l), intersection / l


def resize_cam(cam, size):
    """Map the continuous activation to the original image grid."""
    return np.asarray(Image.fromarray(cam.astype(np.float32)).resize(size, Image.Resampling.BILINEAR))


def threshold_cam(cam, size):
    """Resize normalized continuous activation and apply the fixed cutoff."""
    return resize_cam(cam, size) >= 0.5


def comparison_codes(active, lesion):
    """Encode background, activation only, lesion only, and overlap."""
    return active.astype(np.uint8) + 2 * lesion.astype(np.uint8)


def save_review(image, full_cam, active, lesion, row):
    """Save a labeled visual review of the exact arrays used for metrics."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    present = row['masks'].split(';')
    missing = [suffix[1:-4] for _, suffix in LESION_CATEGORIES if suffix[1:-4] not in present]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.subplots_adjust(left=0.03, right=0.97, bottom=0.10, top=0.88, wspace=0.04, hspace=0.16)
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Grading test image', fontsize=16)
    axes[0, 1].imshow(image)
    axes[0, 1].imshow(full_cam, cmap='magma', vmin=0, vmax=1, alpha=0.55, interpolation='nearest')
    axes[0, 1].set_title(f"Predicted-class Grad-CAM | predicted grade {row['predicted_grade']} | reference grade {row['true_grade']}", fontsize=14)
    axes[1, 0].imshow(lesion, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    axes[1, 0].set_title('Available recorded lesion union (white)', fontsize=16)
    colors = ['#101010', '#168bd2', '#f08020', '#f7e34d']
    axes[1, 1].imshow(comparison_codes(active, lesion), cmap=ListedColormap(colors), vmin=0, vmax=3, interpolation='nearest')
    axes[1, 1].set_title('Thresholded Grad-CAM and recorded lesions', fontsize=16)
    fig.legend(handles=[Patch(color=colors[1], label='Grad-CAM only (≥ 0.5)'),
                        Patch(color=colors[2], label='Recorded lesion only'),
                        Patch(color=colors[3], label='Overlap')],
               loc='lower center', bbox_to_anchor=(0.5, 0.025), ncol=3, fontsize=12)
    for ax in axes.flat:
        ax.axis('off')
    missing_text = ', '.join(missing) if missing else 'none'
    fig.suptitle(f"{row['segmentation_id']} / {row['grading_test_id']} | available: {', '.join(present)} | missing files: {missing_text}\n"
                 'Missing files are unknown, not evidence of lesion absence.', fontsize=16, y=0.97)
    path = OUT / f"{row['segmentation_id']}__{row['grading_test_id']}_review.png"
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f'Visual review: {path}')


def complete_file_mean(rows):
    complete = [row for row in rows if row['complete_files']]
    return {key: float(np.mean([row[key] for row in complete]))
            for key in ('iou', 'dice', 'lesion_coverage')} if complete else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figures', action='store_true', help='Save labeled visual reviews')
    args = parser.parse_args()
    model = load_model()
    rows = []
    for seg, grade in PAIRS:
        seg_id, grade_id = f'IDRiD_{seg}', f'IDRiD_{grade}'
        with Image.open(SEG / f'{seg_id}.jpg') as im:
            size = im.size
        with Image.open(GRADE / f'{grade_id}.jpg') as im:
            image = im.convert('RGB')
        if image.size != size:
            raise ValueError(f'Image dimensions differ for {seg_id}')
        lesion = np.zeros((size[1], size[0]), dtype=bool)
        present = []
        for folder, suffix in LESION_CATEGORIES:
            path = MASK_ROOT / folder / f'{seg_id}{suffix}'
            if not path.is_file():
                continue
            with Image.open(path) as im:
                mask = np.asarray(im)
            if mask.shape != lesion.shape or not np.isin(mask, [0, 1]).all():
                raise ValueError(f'Invalid mask: {path}')
            lesion |= mask == 1
            present.append(suffix[1:-4])
        if not present:
            raise ValueError(f'No masks for {seg_id}')
        if not lesion.any():
            print(f'{seg_id}: excluded: empty recorded lesion union')
            continue
        cam, predicted, confidence = generate_gradcam(model, INFERENCE_TRANSFORM(image).unsqueeze(0).to(DEVICE), model.features[-1])
        full_cam = resize_cam(cam, size)
        active = full_cam >= 0.5
        iou, dice, coverage = overlap(active, lesion)
        rows.append(dict(segmentation_id=seg_id, grading_test_id=grade_id, true_grade=2,
                         predicted_grade=predicted, confidence=confidence, masks=';'.join(present),
                         complete_files=len(present)==4, lesion_pixels=int(lesion.sum()),
                         activation_pixels=int(active.sum()), iou=iou, dice=dice, lesion_coverage=coverage))
        if args.figures:
            save_review(image, full_cam, active, lesion, rows[-1])
    if not rows:
        raise ValueError('No eligible images')
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'per_image.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    for row in rows:
        print(row)
    mean = complete_file_mean(rows)
    if mean is not None:
        print('Complete-file subset mean:', mean, 'n=', sum(r['complete_files'] for r in rows))


if __name__ == '__main__':
    main()
