"""Rebuild the frozen cohort from separately preserved private review decisions."""
import csv
import hashlib
from pathlib import Path

review = Path('outputs/explainability/e_ophtha_review')
decisions = review / 'decisions_v2.csv'
expected = 'e5588f0fe4ce168fd1ee047341a2188c43b0855f6523817f5e66f0b3328c4a85'
if hashlib.sha256(decisions.read_bytes()).hexdigest() != expected:
    raise ValueError('Private reviewed decisions differ from the frozen record')
with decisions.open(newline='') as f:
    rows = list(csv.DictReader(f))
with Path('outputs/explainability/e_ophtha_inventory/image_inventory.csv').open(newline='') as f:
    inventory = list(csv.DictReader(f))
lookup = {(r['subset'], r['image']): r for r in inventory}
hashes = {lookup[(r['external_subset'], r['external_image'])]['file_sha256'] for r in rows}
for row in inventory:
    row['cohort'] = 'unresolved' if row['file_sha256'] in hashes else 'primary'
with (review / 'cohort_v2.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=inventory[0])
    writer.writeheader()
    writer.writerows(inventory)
print('Wrote cohort from private decisions; unresolved unique images:', len(hashes))
