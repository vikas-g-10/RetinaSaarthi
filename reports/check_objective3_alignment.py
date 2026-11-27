"""Check sampled lesion-centered patches; write reproducible evidence separately."""
import csv
import hashlib
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEG = ROOT / 'datasets/idrid/segmentation/extracted/A. Segmentation'
GRADE = ROOT / 'datasets/idrid/disease_grading/extracted/B. Disease Grading/1. Original Images/b. Testing Set'
OUT = ROOT / 'outputs/explainability/idrid_overlap/alignment_patches.csv'
PAIRS = [('09', '091'), ('22', '088'), ('31', '089')]
RADIUS, PAD = 32, 4


def fits_patch(cx, cy, width, height, radius=RADIUS, pad=PAD):
    """Whether the whole search window fits, including exact edge contact."""
    margin = radius + pad
    return cx >= margin and cy >= margin and cx + margin <= width and cy + margin <= height


def read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f'Cannot read {path}')
    return image


def main():
    rows = []
    for a, b in PAIRS:
        paths = [SEG / '1. Original Images/a. Training Set' / f'IDRiD_{a}.jpg',
                 GRADE / f'IDRiD_{b}.jpg',
                 SEG / '2. All Segmentation Groundtruths/a. Training Set/1. Microaneurysms' / f'IDRiD_{a}_MA.tif']
        x, y, mask = [read_gray(p) for p in paths]
        if x.shape != y.shape or x.shape != mask.shape:
            raise ValueError(f'Image or mask dimensions differ for {a}/{b}')
        hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
        _, _, stats, cents = cv2.connectedComponentsWithStats((mask > 0).astype('uint8'))
        # Stable tie-break by component ID, retaining the 30 largest components.
        selected = sorted(range(1, len(stats)), key=lambda i: (int(stats[i, 4]), i))[-30:]
        pair_rows = []
        for i in selected:
            cx, cy = np.rint(cents[i]).astype(int)
            row = dict(segmentation_id=a, grading_test_id=b, component_id=i,
                       component_pixels=int(stats[i, 4]), center_x=int(cx), center_y=int(cy),
                       radius=RADIUS, search_padding=PAD, seg_sha256=hashes[0],
                       grade_sha256=hashes[1], mask_sha256=hashes[2],
                       status='boundary', offset_x='', offset_y='', correlation='')
            if fits_patch(cx, cy, x.shape[1], x.shape[0]):
                t = x[cy-RADIUS:cy+RADIUS, cx-RADIUS:cx+RADIUS]
                search = y[cy-RADIUS-PAD:cy+RADIUS+PAD, cx-RADIUS-PAD:cx+RADIUS+PAD]
                _, score, _, loc = cv2.minMaxLoc(cv2.matchTemplate(search, t, cv2.TM_CCOEFF_NORMED))
                row.update(status='matched', offset_x=loc[0]-PAD, offset_y=loc[1]-PAD, correlation=score)
            pair_rows.append(row)
        rows.extend(pair_rows)
        matched = [row for row in pair_rows if row['status'] == 'matched']
        zero = sum(row['offset_x'] == 0 and row['offset_y'] == 0 for row in matched)
        print(f'{a}/{b}: selected={len(pair_rows)} matched={len(matched)} boundary_skipped={len(pair_rows)-len(matched)} zero_offset={zero}')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'Patch evidence: {OUT}')


if __name__ == '__main__':
    main()
