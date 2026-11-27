"""Reproduce the frozen e-ophtha v2 exploratory analysis."""
import csv
import hashlib
import json
import platform
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torchvision

from ml.explainability.generate_gradcam import (ROOT, CHECKPOINT_PATH, DEVICE,
    INFERENCE_TRANSFORM, generate_gradcam, load_model)
from ml.explainability.evaluate_idrid_overlap import overlap, resize_cam

REVIEW = ROOT / 'outputs/explainability/e_ophtha_review'
OUT = ROOT / 'outputs/explainability/e_ophtha_evaluation_v2'
COHORT_SHA = '5421005fcab4cda3c527618f4b04b7460703adb08b75a8e644bb637cc1fbbac0'
CHECKPOINT_SHA = '11917bd5cc3669ddecfd9d93fd03b86903520689b809cf6c0381cfabe7dfb64c'
PROTOCOL_SHA = '6c285848365409e8408b3f537579258fcb895c0139a50c6927906efc3d4f3ef4'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mask_for_rule(mask, rule):
    if rule == 'nonzero':
        return mask > 0
    if rule == '255_only':
        return mask == 255
    if rule == 'exclude_239_247':
        return (mask > 0) & (mask != 239) & (mask != 247)
    raise ValueError(rule)


def measure(cam, mask, threshold):
    active = cam >= threshold
    area = int(active.sum())
    result = {'active_pixels': area, 'image_pixels': int(active.size),
              'active_fraction': area / active.size}
    if mask is not None:
        count = int(mask.sum())
        result.update(mask_pixels=count, intersection_pixels=int(np.count_nonzero(active & mask)))
        if count:
            result.update(zip(('iou', 'dice', 'coverage'), overlap(active, mask)))
    return result


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def summarize(rows):
    groups = defaultdict(list)
    for r in rows:
        for cohort in (r['cohort'], 'combined'):
            groups[(r['subset'], r['category'], cohort, r['rule'], r['threshold'])].append(r)
    result = []
    for (subset, category, cohort, rule, threshold), rs in sorted(groups.items()):
        row = dict(subset=subset, category=category, cohort=cohort, rule=rule,
                   threshold=threshold, total=len(rs))
        for metric in (('iou', 'dice', 'coverage') if category == 'positive' else ('active_fraction',)):
            values = [r[metric] for r in rs if metric in r]
            row['valid'] = len(values)
            row['excluded_empty_mask'] = len(rs) - len(values)
            if values:
                row[metric + '_mean'] = statistics.mean(values)
                row[metric + '_median'] = statistics.median(values)
        if category == 'negative':
            row['fraction_any_active'] = sum(r['active_pixels'] > 0 for r in rs) / len(rs)
        result.append(row)
    return result


def evaluate(model, output_dir, protocol_sha, checkpoint_sha):
    """Evaluate one already verified model with the frozen cohort and metrics."""
    torch.set_num_threads(2)
    with (REVIEW / 'cohort_v2.csv').open() as f:
        inventory = list(csv.DictReader(f))
    assert len(inventory) == 463
    results = []
    cache = {}
    for i, item in enumerate(inventory, 1):
        subset = item['subset']
        if subset not in ('EX', 'MA') or item['source_category'] not in (subset, 'healthy'):
            raise ValueError(f'Unexpected source category: {item}')
        if item['cohort'] not in ('primary', 'unresolved'):
            raise ValueError(f'Unexpected cohort: {item}')
        base = ROOT / 'datasets/e_ophtha' / subset.lower() / ('e_optha_' + subset)
        path = base / item['image']
        if digest(path) != item['file_sha256']:
            raise ValueError(f'Image hash differs: {path}')
        with Image.open(path) as im:
            im = im.convert('RGB')
            size = im.size
            if size != (int(item['width']), int(item['height'])):
                raise ValueError(f'Image dimensions differ: {path}')
            if item['file_sha256'] not in cache:
                tensor = INFERENCE_TRANSFORM(im).unsqueeze(0).to(DEVICE)
                cache[item['file_sha256']] = generate_gradcam(model, tensor, model.features[-1])
        cam, predicted, confidence = cache[item['file_sha256']]
        full = resize_cam(cam, size)
        category = 'negative' if item['source_category'] == 'healthy' else 'positive'
        mask = None
        if category == 'positive':
            if item['status'] != 'matched' or item['mask_count'] != '1':
                raise ValueError(f'Unexpected mask status: {path}')
            with Image.open(base / item['masks']) as m:
                mask = np.asarray(m)
            if mask.shape != (size[1], size[0]) or not np.isin(mask, [0, 200, 239, 247, 255]).all() or not mask.any():
                raise ValueError(f'Unexpected mask: {path}')
        elif item['mask_count'] != '0':
            raise ValueError(f'Unexpected source-negative mask: {path}')
        mask_sha256 = digest(base / item['masks']) if mask is not None else ''
        scenarios = [('nonzero', t) for t in (0.25, 0.5, 0.75)]
        if subset == 'MA' and category == 'positive':
            scenarios += [('255_only', 0.5), ('exclude_239_247', 0.5)]
        for rule, threshold in scenarios:
            binary = None if mask is None else mask_for_rule(mask, rule)
            row = dict(subset=subset, category=category, cohort=item['cohort'], image=item['image'],
                       image_sha256=item['file_sha256'], predicted_grade=predicted,
                       mask_sha256=mask_sha256, confidence=confidence, rule=rule, threshold=threshold)
            row.update(measure(full, binary, threshold))
            results.append(row)
        if i % 25 == 0:
            print(f'Processed {i}/{len(inventory)} images', flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / 'per_image.csv', results)
    write_csv(output_dir / 'aggregate.csv', summarize(results))
    (output_dir / 'run_metadata.json').write_text(json.dumps(dict(protocol_sha256=protocol_sha,
        cohort_sha256=COHORT_SHA, checkpoint_sha256=checkpoint_sha, device=str(DEVICE),
        python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__,
        numpy=np.__version__, unique_images=len(cache)), indent=2) + '\n')
    print(f'Wrote {len(results)} rows from {len(cache)} unique images to {output_dir}')


def main():
    for path, expected in [(REVIEW / 'cohort_v2.csv', COHORT_SHA),
                           (CHECKPOINT_PATH, CHECKPOINT_SHA),
                           (ROOT / 'reports/e_ophtha_protocol_v2.md', PROTOCOL_SHA)]:
        if digest(path) != expected:
            raise ValueError(f'Frozen input hash mismatch: {path}')
    evaluate(load_model(), OUT, PROTOCOL_SHA, CHECKPOINT_SHA)


if __name__ == '__main__':
    main()
