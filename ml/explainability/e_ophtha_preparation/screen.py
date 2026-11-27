"""Conservative fixed-grid duplicate screen; never modify images or training inputs."""
import csv, hashlib, json
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'outputs/explainability/e_ophtha_review'
rows = list(csv.DictReader((ROOT/'outputs/explainability/e_ophtha_inventory/image_inventory.csv').open()))
training = list(csv.DictReader((ROOT/'outputs/classification/splits/combined_train_aptos_idrid.csv').open()))
external = {}
for r in rows:
    external.setdefault(r['file_sha256'], r)
external = list(external.values())
base = ROOT/'datasets/e_ophtha'
extpaths = [base/r['subset'].lower()/f"e_optha_{r['subset']}"/r['image'] for r in external]

def features(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f'Unreadable image: {path}')
    small = cv2.resize(img[:,:,1], (64,64), interpolation=cv2.INTER_AREA)
    small = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8)).apply(small)
    small = cv2.GaussianBlur(small,(5,5),0).astype(np.float32)
    small -= small.mean()
    norm = np.linalg.norm(small)
    if not norm: raise ValueError(f'Constant feature: {path}')
    hashimage = cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),(9,8),interpolation=cv2.INTER_AREA)
    dhash = (hashimage[:,1:] > hashimage[:,:-1]).reshape(-1)
    return small.ravel()/norm, dhash

print('external unique images',len(external),'manifest training images',len(training),flush=True)
efeat, ehash = zip(*(features(p) for p in extpaths))
efeat=np.stack(efeat); ehash=np.stack(ehash)
filehash={r['file_sha256']:r for r in external}
pixelhash={(int(r['width']),int(r['height']),r['decoded_rgb_sha256']):r for r in external}
sizes={(x[0],x[1]) for x in pixelhash}
trainfeatures=[]; trainhashes=[]; exact=[]
for i,r in enumerate(training):
    path=ROOT/r['image_path']
    f,h=features(path); trainfeatures.append(f); trainhashes.append(h)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if digest in filehash: exact.append({'type':'byte','external':filehash[digest]['image'],'train':r['image_path']})
    with Image.open(path) as im:
        if im.size in sizes:
            rgb=np.asarray(im.convert('RGB')); pd=hashlib.sha256(f'{im.size}:'.encode()+rgb.tobytes()).hexdigest()
            hit=pixelhash.get((im.width,im.height,pd))
            if hit: exact.append({'type':'decoded_rgb','external':hit['image'],'train':r['image_path']})
    if (i+1)%500==0: print('processed',i+1,flush=True)
trainfeatures=np.stack(trainfeatures); trainhashes=np.stack(trainhashes)
scores=efeat @ trainfeatures.T
pairs=[]
for i in range(len(external)):
    dist=np.count_nonzero(trainhashes != ehash[i],axis=1)
    indexes=np.flatnonzero((scores[i]>=0.97) | (dist<=4))
    for j in indexes:
        pairs.append(dict(external_subset=external[i]['subset'],external_image=external[i]['image'],train_source=training[j]['source'],train_image=training[j]['image_path'],green_correlation=float(scores[i,j]),dhash_distance=int(dist[j])))
pairs.sort(key=lambda r:(-r['green_correlation'],r['dhash_distance']))
fields=['external_subset','external_image','train_source','train_image','green_correlation','dhash_distance']
with (OUT/'duplicate_candidates.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(pairs)
(OUT/'screen_summary.json').write_text(json.dumps({'external_unique':len(external),'manifest_rows':len(training),'external_rows':len(rows),'correlation_threshold':0.97,'dhash_distance_threshold':4,'candidate_pairs':len(pairs),'candidate_external_images':len({(x['external_subset'],x['external_image']) for x in pairs}),'max_correlation':float(scores.max()),'exact_matches':exact},indent=2)+'\n')
print('candidate pairs',len(pairs),'max correlation',float(scores.max()),'exact',len(exact),flush=True)
