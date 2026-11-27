import csv, hashlib
from pathlib import Path
root=Path('.')
out=root/'outputs/classification/replacement_eophtha_v1'
def read(p):
    with open(p,newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
train=read('outputs/classification/splits/combined_train_aptos_idrid.csv')
decisions=read('outputs/explainability/e_ophtha_review/decisions_v2.csv')
paths={r['train_image'] for r in decisions}
matched=[r for r in train if r['image_path'] in paths]
assert len(paths)==len(matched)==22
badgroups={r['duplicate_group'] for r in matched}
excluded=[r for r in train if r['duplicate_group'] in badgroups]
retained=[r for r in train if r['duplicate_group'] not in badgroups]
write(out/'provisional_train.csv',retained)
write(out/'review_exclusions.csv',[dict(r,reason='unresolved e-ophtha review or same recorded duplicate group') for r in excluded])
splits=read('outputs/classification/splits/aptos_grouped_seed42.csv')
write(out/'provisional_validation.csv',[r for r in splits if r['split']=='validation'])
for name in ['provisional_train.csv','review_exclusions.csv','provisional_validation.csv']:
 p=out/name;print(name,hashlib.sha256(p.read_bytes()).hexdigest())
print('retained',len(retained),'excluded',len(excluded),'provisional validation',sum(r['split']=='validation' for r in splits))
