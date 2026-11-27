"""Apply the established two-stage duplicate screen to APTOS validation."""
import csv, hashlib, importlib.util, json
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
from PIL import Image
ROOT=Path.cwd(); OUT=ROOT/'outputs/classification/replacement_eophtha_v1'; REVIEW=ROOT/'outputs/explainability/e_ophtha_review'
spec=importlib.util.spec_from_file_location('stage2',Path(__file__).with_name('second_stage.py')); stage2=importlib.util.module_from_spec(spec); spec.loader.exec_module(stage2)
def readcsv(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def writecsv(p,rows,fields):
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def features(path):
    img=cv2.imread(str(path),cv2.IMREAD_COLOR)
    if img is None: raise ValueError(path)
    small=cv2.resize(img[:,:,1],(64,64),interpolation=cv2.INTER_AREA)
    small=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8)).apply(small)
    small=cv2.GaussianBlur(small,(5,5),0).astype(np.float32);small-=small.mean()
    norm=np.linalg.norm(small)
    if not norm: raise ValueError(path)
    h=cv2.resize(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),(9,8),interpolation=cv2.INTER_AREA)
    return small.ravel()/norm,(h[:,1:]>h[:,:-1]).reshape(-1)
inv=readcsv(ROOT/'outputs/explainability/e_ophtha_inventory/image_inventory.csv'); ext={}
for x in inv:ext.setdefault(x['file_sha256'],x)
ext=list(ext.values()); val=readcsv(OUT/'provisional_validation.csv')
assert len(ext)==434 and len(val)==541
paths=[ROOT/'datasets/e_ophtha'/r['subset'].lower()/f"e_optha_{r['subset']}"/r['image'] for r in ext]
vp=[ROOT/'datasets/aptos/train_images'/f"{r['id_code']}.png" for r in val]
cv2.setNumThreads(1)
ef,eh=zip(*(features(p) for p in paths)); ef=np.stack(ef);eh=np.stack(eh)
vf=[];vh=[];exact=[]
filehash={r['file_sha256']:r for r in ext}; pixhash={(int(r['width']),int(r['height']),r['decoded_rgb_sha256']):r for r in ext}; sizes={(k[0],k[1]) for k in pixhash}
for i,p in enumerate(vp):
 f,h=features(p);vf.append(f);vh.append(h); sha=hashlib.sha256(p.read_bytes()).hexdigest()
 if sha in filehash:exact.append(dict(type='byte',external=filehash[sha]['image'],validation=val[i]['id_code']))
 with Image.open(p) as im:
  if im.size in sizes:
   rgb=np.asarray(im.convert('RGB')); digest=hashlib.sha256(f'{im.size}:'.encode()+rgb.tobytes()).hexdigest(); x=pixhash.get((im.width,im.height,digest))
   if x:exact.append(dict(type='decoded_rgb',external=x['image'],validation=val[i]['id_code']))
 if (i+1)%100==0: print('validation features',i+1,flush=True)
vf=np.stack(vf);vh=np.stack(vh); scores=ef@vf.T;pairs=[]
for i,x in enumerate(ext):
 distances=np.count_nonzero(vh!=eh[i],axis=1)
 for j in np.flatnonzero((scores[i]>=.97)|(distances<=4)):
  pairs.append(dict(external_subset=x['subset'],external_image=x['image'],validation_id=val[j]['id_code'],green_correlation=float(scores[i,j]),dhash_distance=int(distances[j])))
pairs.sort(key=lambda x:(-x['green_correlation'],x['dhash_distance']))
fields=['external_subset','external_image','validation_id','green_correlation','dhash_distance'];writecsv(OUT/'validation_first_stage.csv',pairs,fields)
pathmap={(r['subset'],r['image']):p for r,p in zip(ext,paths)};vmap={r['id_code']:p for r,p in zip(val,vp)};cache={};res=[]
for i,r in enumerate(pairs):
 a=pathmap[(r['external_subset'],r['external_image'])];b=vmap[r['validation_id']]
 for p in (a,b):
  if p not in cache:cache[p]=stage2.feature(stage2.read(p))
 m,n,ratio,label=stage2.compare(cache[a],cache[b]);res.append(dict(**r,orb_matches=m,affine_inliers=n,inlier_ratio=ratio,screen_status=label))
 if (i+1)%500==0:print('second stage',i+1,'/',len(pairs),flush=True)
fields+=['orb_matches','affine_inliers','inlier_ratio','screen_status'];writecsv(OUT/'validation_second_stage.csv',res,fields)
short=[x for x in res if x['screen_status']!='weak'];short.sort(key=lambda r:(-r['affine_inliers'],-r['inlier_ratio']));writecsv(OUT/'validation_review.csv',short,fields)
(OUT/'validation_screen_summary.json').write_text(json.dumps(dict(external_unique=len(ext),validation=len(val),pairs_tested=len(ext)*len(val),exact=exact,first_stage=len(pairs),second_stage=dict(Counter(x['screen_status'] for x in res)),max_correlation=float(scores.max())),indent=2)+'\n')
print('DONE',len(pairs),Counter(x['screen_status'] for x in res),'exact',len(exact),flush=True)
