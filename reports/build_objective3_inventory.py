"""Inventory mapping evidence; no inference, registration, or model evaluation."""
import csv
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports'
SEG = ROOT / 'datasets/idrid/segmentation/extracted/A. Segmentation'
GRADE = ROOT / 'datasets/idrid/disease_grading/extracted/B. Disease Grading'
MANIFEST = ROOT / 'outputs/classification/splits/combined_train_aptos_idrid.csv'
OUTPUT = REPORT / 'objective3_candidate_inventory.csv'
CATEGORIES = {'MA': '1. Microaneurysms', 'HE': '2. Haemorrhages',
              'EX': '3. Hard Exudates', 'SE': '4. Soft Exudates'}
SPLITS = {'train': 'a. Training Set', 'test': 'b. Testing Set'}


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def size(path):
    if not path.is_file():
        return None
    with Image.open(path) as image:
        return image.size


def main():
    mapping = read_csv(REPORT / 'idrid_mapping/mapping.csv')
    eligibility = {(r['seg_split'], r['seg_file']): r for r in read_csv(REPORT / 'idrid_eligibility.csv')}
    duplicates = {(r['split'], r['filename']): r for r in read_csv(REPORT / 'idrid_grading_duplicates.csv')}
    manifest = {r['image_path'] for r in read_csv(MANIFEST) if r['source'].upper() == 'IDRID'}
    result = []
    for m in mapping:
        key = m['seg_split'], m['seg_file']
        e = eligibility[key]
        gs, gf = m['grade_split'], m['grade_file']
        seg_path = SEG / '1. Original Images' / SPLITS[key[0]] / key[1]
        grade_path = GRADE / '1. Original Images' / SPLITS[gs] / gf
        seg_size, grade_size = size(seg_path), size(grade_path)
        available, missing, bad_size = [], [], []
        for label, folder in CATEGORIES.items():
            path = SEG / '2. All Segmentation Groundtruths' / SPLITS[key[0]] / folder / f'{seg_path.stem}_{label}.tif'
            mask_size = size(path)
            if mask_size is None:
                missing.append(label)
            else:
                available.append(label)
                if mask_size != seg_size:
                    bad_size.append(label)
        dup = duplicates.get((gs, gf))
        in_manifest = grade_path.relative_to(ROOT).as_posix() in manifest
        reasons = []
        if m['status'] != 'confident':
            reasons.append('mapping uncertain; best match is not verified')
        if e['pairing_status'] in ('tie_duplicate', 'tie_label_conflict'):
            reasons.append(e['notes'].split(' Boolean')[0])
        if in_manifest:
            reasons.append('grading partner in current training manifest')
        if dup:
            reasons.append(f"duplicate audit group {dup['group_id']}; label disagreement={dup['label_disagreement']}")
        if not seg_size or not grade_size:
            reasons.append('image missing')
        elif seg_size != grade_size:
            reasons.append('image dimensions differ')
        if bad_size:
            reasons.append('mask dimensions differ: ' + ','.join(bad_size))
        if not available:
            reasons.append('no available lesion masks')
        if missing:
            reasons.append('missing mask files (unknown): ' + ','.join(missing))
        if m['status'] == 'confident' and gs == 'test' and not dup and not in_manifest and seg_size and seg_size == grade_size and available and not bad_size:
            disposition = 'existing exploratory candidate'
            reasons.append('sampled local alignment supports zero shift; independence unproven')
        elif in_manifest and m['status'] == 'confident':
            disposition = 'exclude: current-manifest exposure'
            reasons.append('checkpoint manifest provenance unverified')
        else:
            disposition = 'uncertain / not eligible under current protocol'
            reasons.append('alignment and checkpoint/patient independence unverified')
        if key == ('test', 'IDRiD_68.jpg'):
            reasons.append('visual disc/vessel comparison with 034 does not establish correspondence; retain exclusion')
        elif key == ('test', 'IDRiD_70.jpg'):
            reasons.append('70/006 likely same eye, possibly different capture; sampled lesion neighborhoods are displaced; retain exclusion')
        result.append(dict(seg_split=key[0], seg_file=key[1], best_grade_split=gs, best_grade_file=gf,
                           eligibility_status=e['pairing_status'], grade=e['dr_grade'], mapping_status=m['status'],
                           score=m['score'], margin=m['margin'], shift_at_width_256=m['shift_px_at256'],
                           image_dimensions_match=bool(seg_size and seg_size == grade_size),
                           available_masks=';'.join(available), missing_mask_files=';'.join(missing),
                           mask_dimension_mismatches=';'.join(bad_size), duplicate_group=dup['group_id'] if dup else '',
                           current_training_manifest_path=in_manifest, disposition=disposition, reasons='; '.join(reasons)))
    if len(result) != len(eligibility) or len(result) != 81:
        raise ValueError('Unexpected mapping or eligibility row count')
    with OUTPUT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=result[0].keys(), lineterminator='\n'); writer.writeheader(); writer.writerows(result)
    print('rows:', len(result), 'dispositions:', dict(Counter(r['disposition'] for r in result)))
    print('grading-test best matches:')
    for r in result:
        if r['best_grade_split'] == 'test':
            print(r['seg_split'], r['seg_file'], '->', r['best_grade_file'], r['disposition'], r['reasons'])
    print('saved', OUTPUT)


if __name__ == '__main__':
    main()
