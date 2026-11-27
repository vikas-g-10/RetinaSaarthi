"""Read-only image/mask inventory; output contains metadata, not source images."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/explainability/e_ophtha_inventory'
BASE = ROOT / 'datasets/e_ophtha'
all_rows = []
summaries = {}
for subset in ('EX', 'MA'):
    root = BASE / subset.lower() / f'e_optha_{subset}'
    images = sorted(p for folder in (subset, 'healthy') for p in (root / folder).rglob('*') if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg'))
    masks = sorted(p for p in (root / f'Annotation_{subset}').rglob('*') if p.is_file() and p.suffix.lower() == '.png')
    by_key = defaultdict(list)
    for p in masks:
        name = p.stem
        if subset == 'EX' and name.endswith('_EX'):
            name = name[:-3]
        by_key[(p.parent.name, name.casefold())].append(p)
    seen = set()
    rows = []
    for image in images:
        category = image.relative_to(root).parts[0]
        key = image.parent.name, image.stem.casefold()
        matched = by_key.get(key, [])
        with Image.open(image) as im:
            rgb = np.asarray(im.convert('RGB'))
            size = im.size
            pixel_hash = hashlib.sha256(f'{size}:'.encode() + rgb.tobytes()).hexdigest()
        row = {'subset': subset, 'source_category': category, 'visit': key[0], 'image': image.relative_to(root).as_posix(),
               'width': size[0], 'height': size[1], 'file_sha256': hashlib.sha256(image.read_bytes()).hexdigest(),
               'decoded_rgb_sha256': pixel_hash, 'mask_count': len(matched), 'masks': ';'.join(p.relative_to(root).as_posix() for p in matched),
               'mask_dimensions': '', 'mask_values': '', 'mask_nonzero_pixels': '', 'status': ''}
        if len(matched) == 1:
            mask = matched[0]
            seen.add(mask)
            with Image.open(mask) as im:
                arr = np.asarray(im)
                row['mask_dimensions'] = f'{im.width}x{im.height}'
                row['mask_values'] = ';'.join(str(v) for v in np.unique(arr).tolist())
                row['mask_nonzero_pixels'] = int(np.count_nonzero(arr))
                row['status'] = 'matched' if im.size == size and arr.ndim == 2 else 'mask_shape_mismatch'
        elif len(matched) > 1:
            seen.update(matched)
            row['status'] = 'ambiguous_multiple_masks'
        else:
            row['status'] = 'no_mask_positive' if category == subset else 'no_mask_source_healthy'
        rows.append(row)
    unpaired = [p.relative_to(root).as_posix() for p in masks if p not in seen]
    other = Counter(p.suffix.lower() or '<none>' for p in root.rglob('*') if p.is_file() and p not in images and p not in masks)
    summaries[subset] = {'images': len(images), 'masks': len(masks), 'categories': dict(Counter(r['source_category'] for r in rows)),
                         'statuses': dict(Counter(r['status'] for r in rows)), 'mask_values': dict(Counter(r['mask_values'] for r in rows if r['mask_values'])),
                         'image_dimensions': dict(Counter(f"{r['width']}x{r['height']}" for r in rows)),
                         'zero_masks': sum(r['mask_nonzero_pixels'] == 0 for r in rows), 'unmatched_masks': unpaired, 'other_files': dict(other),
                         'visits': len({r['visit'] for r in rows})}
    all_rows.extend(rows)
    print(subset, json.dumps(summaries[subset]), flush=True)
for field in ('file_sha256', 'decoded_rgb_sha256'):
    a = {r[field] for r in all_rows if r['subset'] == 'EX'}
    b = {r[field] for r in all_rows if r['subset'] == 'MA'}
    shared = a & b
    summaries['overlap_' + field] = {'unique_hashes': len(shared), 'EX_images': sum(r[field] in shared for r in all_rows if r['subset']=='EX'), 'MA_images': sum(r[field] in shared for r in all_rows if r['subset']=='MA')}
shared_names = {(r['visit'], Path(r['image']).stem.casefold()) for r in all_rows if r['subset']=='EX'} & {(r['visit'], Path(r['image']).stem.casefold()) for r in all_rows if r['subset']=='MA'}
summaries['shared_visit_image_names'] = len(shared_names)
with (OUT / 'image_inventory.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=all_rows[0].keys(), lineterminator='\n'); writer.writeheader(); writer.writerows(all_rows)
(OUT / 'summary.json').write_text(json.dumps(summaries, indent=2) + '\n')
print('overlap:', json.dumps({k:v for k,v in summaries.items() if k.startswith('overlap') or k.startswith('shared')}), flush=True)
