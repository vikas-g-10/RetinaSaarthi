"""Make image-only comparison sheets; no model inference or registration."""
import json
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SEG = ROOT / 'datasets/idrid/segmentation/extracted/A. Segmentation'
GRADE = ROOT / 'datasets/idrid/disease_grading/extracted/B. Disease Grading/1. Original Images'
OUT = ROOT / 'outputs/explainability/idrid_overlap/candidate_review'
PAIRS = [('68', '034', '381'), ('70', '006', '136')]


def lesion_review():
    """Sample large annotated components and compare unregistered local crops."""
    seg_path = SEG / '1. Original Images/b. Testing Set/IDRiD_70.jpg'
    grade_path = GRADE / 'b. Testing Set/IDRiD_006.jpg'
    x, y = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in (seg_path, grade_path)]
    if x is None or y is None or x.shape != y.shape:
        raise ValueError('Missing or incompatible images')
    h, w = x.shape[:2]
    candidates = []
    for kind, folder in [('MA', '1. Microaneurysms'), ('HE', '2. Haemorrhages'), ('EX', '3. Hard Exudates'), ('SE', '4. Soft Exudates')]:
        path = SEG / '2. All Segmentation Groundtruths/b. Testing Set' / folder / f'IDRiD_70_{kind}.tif'
        with Image.open(path) as image:
            mask = np.asarray(image)
        if mask.shape != (h, w) or not np.isin(mask, [0, 1]).all():
            raise ValueError(f'Invalid mask {path}')
        _, _, stats, centers = cv2.connectedComponentsWithStats((mask > 0).astype('uint8'))
        for i in range(1, len(stats)):
            cx, cy = np.rint(centers[i]).astype(int)
            if 280 <= cx <= w - 280 and 280 <= cy <= h - 280:
                candidates.append((int(stats[i, 4]), kind, i, int(cx), int(cy)))
    selected = []
    for item in sorted(candidates, reverse=True):
        if all((item[3] - v[3]) ** 2 + (item[4] - v[4]) ** 2 >= 400 ** 2 for v in selected):
            selected.append(item)
        if len(selected) == 6:
            break
    if len(selected) < 6:
        raise ValueError('Fewer than six separated lesion samples')
    fig, axes = plt.subplots(len(selected), 2, figsize=(10, 20))
    fig.subplots_adjust(left=0.06, right=0.96, top=0.94, bottom=0.03, wspace=0.08, hspace=0.35)
    rows = []
    for n, (area, kind, component, cx, cy) in enumerate(selected):
        radius, padding = 80, 200
        template = x[cy-radius:cy+radius, cx-radius:cx+radius, 1]
        search = y[cy-radius-padding:cy+radius+padding, cx-radius-padding:cx+radius+padding, 1]
        _, corr, _, loc = cv2.minMaxLoc(cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED))
        dx, dy = loc[0] - padding, loc[1] - padding
        rows.append(dict(sample=n+1, lesion_type=kind, component_id=component, component_pixels=area,
                         center_x=cx, center_y=cy, template_radius=radius, search_padding=padding,
                         best_dx=dx, best_dy=dy, correlation=corr))
        for col, (img, label) in enumerate(((x, 'Segmentation 70'), (y, 'Grading 006'))):
            crop = cv2.cvtColor(img[cy-240:cy+240, cx-240:cx+240], cv2.COLOR_BGR2RGB)
            axes[n, col].imshow(crop)
            axes[n, col].plot(240, 240, '+', color='cyan', markersize=10, markeredgewidth=2)
            axes[n, col].set_title(f'{label} | sample {n+1}: {kind} at ({cx}, {cy})', fontsize=10)
            axes[n, col].axis('off')
    fig.suptitle('70 / 006: annotated-lesion neighborhoods, same native coordinates\nCyan cross = segmentation-mask component centroid; no registration', fontsize=13)
    fig.savefig(OUT / 'IDRiD_70_lesion_neighborhoods.png', dpi=130)
    plt.close(fig)
    with (OUT / 'IDRiD_70_lesion_samples.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys()); writer.writeheader(); writer.writerows(rows)
    for row in rows:
        print('lesion sample:', row)


def crop_bounds(cx, cy, w, h, side):
    side = min(side, w, h)
    left = max(0, min(int(round(cx - side / 2)), w - side))
    top = max(0, min(int(round(cy - side / 2)), h - side))
    return (left, top, left + side, top + side)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    evidence = []
    for seg, best, second in PAIRS:
        paths = [SEG / '1. Original Images/b. Testing Set' / f'IDRiD_{seg}.jpg',
                 GRADE / 'b. Testing Set' / f'IDRiD_{best}.jpg',
                 GRADE / 'a. Training Set' / f'IDRiD_{second}.jpg']
        images = []
        for path in paths:
            with Image.open(path) as im:
                images.append(im.convert('RGB'))
        if len({im.size for im in images}) != 1:
            raise ValueError(f'Dimensions differ for {seg}')
        mask_path = SEG / '2. All Segmentation Groundtruths/b. Testing Set/5. Optic Disc' / f'IDRiD_{seg}_OD.tif'
        with Image.open(mask_path) as im:
            disc = np.asarray(im)
        if disc.shape != (images[0].height, images[0].width) or not np.any(disc):
            raise ValueError(f'Invalid optic disc mask for {seg}')
        ys, xs = np.nonzero(disc)
        cx, cy = float(xs.mean()), float(ys.mean())
        w, h = images[0].size
        disc_box = crop_bounds(cx, cy, w, h, 900)
        # Move from disc towards the image center to show branching vessels.
        vessel_box = crop_bounds(cx + (w / 2 - cx) * 0.55, cy + (h / 2 - cy) * 0.55, w, h, 900)
        labels = [f'Segmentation test IDRiD_{seg}', f'Best: grading test IDRiD_{best}', f'Second: grading train IDRiD_{second}']
        fig, axes = plt.subplots(3, 3, figsize=(17, 13))
        fig.subplots_adjust(left=0.025, right=0.98, bottom=0.04, top=0.90, wspace=0.06, hspace=0.17)
        for col, (im, label) in enumerate(zip(images, labels)):
            axes[0, col].imshow(im)
            axes[0, col].set_title(label, fontsize=13)
            for row, (box, name) in enumerate(((disc_box, 'Optic-disc detail'), (vessel_box, 'Vessel detail')), 1):
                crop = np.asarray(im.crop(box))
                green = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(crop[:, :, 1])
                axes[row, col].imshow(green, cmap='gray', vmin=0, vmax=255)
                axes[row, col].set_title(f'{name} | same coordinates', fontsize=12)
            for row in range(3):
                axes[row, col].axis('off')
        fig.suptitle(f'Candidate comparison: IDRiD_{seg} | original full views; green-channel CLAHE detail\n'
                     'Crops use segmentation-image coordinates; no registration', fontsize=15)
        filename = OUT / f'IDRiD_{seg}_candidate_comparison.png'
        fig.savefig(filename, dpi=140)
        plt.close(fig)
        evidence.append(dict(seg=seg, best=best, second=second, size=[w, h],
                             disc_center=[round(cx, 2), round(cy, 2)], disc_box=disc_box,
                             vessel_box=vessel_box, detail='green channel CLAHE clip=2 tile=8x8',
                             box_convention='left,top inclusive; right,bottom exclusive'))
        print(filename, 'disc_box', disc_box, 'vessel_box', vessel_box)
    (OUT / 'comparison_coordinates.json').write_text(json.dumps(evidence, indent=2) + '\n')
    lesion_review()


if __name__ == '__main__':
    main()
