"""Derive manifests using the separately preserved private validation decisions."""
import collections
import csv
import hashlib
from pathlib import Path

root = Path.cwd()
out = root / 'outputs/classification/replacement_eophtha_v1'
def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))
def write(path, rows, fields):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
decision_path = out / 'validation_decisions.csv'
expected = 'e6d63376896b8b6356930a08bb346ea7d80ee20c5bd3893c7114465c1ab0cb8e'
if hashlib.sha256(decision_path.read_bytes()).hexdigest() != expected:
    raise ValueError('Private validation decisions differ from the frozen record')
reviews = read(decision_path)
val = read(out / 'provisional_validation.csv')
train = read(out / 'provisional_train.csv')
ids = {r['validation_id'] for r in reviews}
groups = {r['duplicate_group'] for r in val if r['id_code'] in ids}
assert len(ids) == 5
removed = [r for r in val if r['duplicate_group'] in groups]
kept = [r for r in val if r['duplicate_group'] not in groups]
train_keep = [r for r in train if r['duplicate_group'] not in {'APTOS:' + g for g in groups}]
for r in kept:
    r.update(source='APTOS', image_path='datasets/aptos/train_images/' + r['id_code'] + '.png', duplicate_group='APTOS:' + r['duplicate_group'])
    r.pop('split')
fields = ['source', 'id_code', 'diagnosis', 'image_path', 'duplicate_group']
write(out / 'final_train.csv', train_keep, fields)
write(out / 'final_validation.csv', kept, fields)
write(out / 'validation_exclusions.csv', removed, list(removed[0]))
assert set(r['duplicate_group'] for r in train_keep).isdisjoint(r['duplicate_group'] for r in kept)
for name, rows in [('train', train_keep), ('validation', kept)]:
    print(name, len(rows), 'grades', dict(sorted(collections.Counter(r['diagnosis'] for r in rows).items())))
