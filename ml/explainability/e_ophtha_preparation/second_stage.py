"""Retina-interior ORB screen with synthetic controls; no model inference."""
import csv, json
from pathlib import Path
from collections import Counter
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/explainability/e_ophtha_review'
cv2.setNumThreads(1)
cv2.setRNGSeed(42)
WIDTH=512
orb=cv2.ORB_create(nfeatures=900, scaleFactor=1.2, nlevels=8, edgeThreshold=20, fastThreshold=8)
matcher=cv2.BFMatcher(cv2.NORM_HAMMING)


def read(path):
    image=cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None: raise ValueError(f'Cannot read {path}')
    h,w=image.shape[:2]
    return cv2.resize(image,(WIDTH,round(h*WIDTH/w)),interpolation=cv2.INTER_AREA)


def feature(image):
    green=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8)).apply(image[:,:,1])
    valid=(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)>20).astype('uint8')*255
    valid=cv2.erode(valid,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(21,21)))
    keypoints,desc=orb.detectAndCompute(green,valid)
    return np.array([k.pt for k in keypoints],dtype='float32'),desc


def compare(a,b):
    pa,da=a; pb,db=b
    if da is None or db is None or len(da)<2 or len(db)<2: return (0,0,0.0,'insufficient')
    raw=matcher.knnMatch(da,db,k=2)
    good=[x[0] for x in raw if len(x)==2 and x[0].distance < .75*x[1].distance]
    if len(good)<3: return (len(good),0,0.0,'weak')
    src=np.array([pa[m.queryIdx] for m in good]); dst=np.array([pb[m.trainIdx] for m in good])
    _,inliers=cv2.estimateAffinePartial2D(src,dst,method=cv2.RANSAC,ransacReprojThreshold=4,maxIters=1000,confidence=.99,refineIters=10)
    n=int(inliers.sum()) if inliers is not None else 0
    ratio=n/len(good)
    label='review' if n>=8 and ratio>=.35 else 'weak'
    return len(good),n,ratio,label


def main():
    pairs=list(csv.DictReader((OUT/'duplicate_candidates.csv').open()))
    manifest=list(csv.DictReader((ROOT/'outputs/classification/splits/combined_train_aptos_idrid.csv').open()))
    # Controls are drawn only from current training data; no e-ophtha images.
    controls=[ROOT/r['image_path'] for r in manifest[:12]]
    images=[read(p) for p in controls]
    rows=[]
    for i,image in enumerate(images):
        h,w=image.shape[:2]
        jpeg=cv2.imdecode(cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,45])[1],cv2.IMREAD_COLOR)
        shifted=cv2.warpAffine(image,np.float32([[1,0,15],[0,1,-10]]),(w,h))
        crop=image[int(h*.08):int(h*.92),int(w*.08):int(w*.92)]
        small=cv2.resize(cv2.resize(image,(w//2,h//2)),(w,h))
        for kind,other,expected in [('jpeg45',jpeg,True),('shift_15_-10',shifted,True),('crop_8_percent',crop,True),('downscale_half',small,True),('unrelated',images[(i+1)%len(images)],False)]:
            m,n,ratio,label=compare(feature(image),feature(other))
            rows.append(dict(control_image=str(controls[i].relative_to(ROOT)),transformation=kind,expected_match=expected,matches=m,inliers=n,inlier_ratio=ratio,classification=label))
    # The historical run also used one privately documented mapped-image control.
    with (OUT/'second_stage_controls.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    print('controls',dict(Counter((x['transformation'],x['classification']) for x in rows)),flush=True)
    # Cache only images present in first-stage pairs.
    paths={}
    for r in pairs:
        ep=ROOT/'datasets/e_ophtha'/r['external_subset'].lower()/f"e_optha_{r['external_subset']}"/r['external_image']
        paths[str(ep)]=ep; tp=ROOT/r['train_image']; paths[str(tp)]=tp
    cache={}
    for i,(key,p) in enumerate(paths.items()):
        cache[key]=feature(read(p))
        if (i+1)%500==0: print('features',i+1,'/',len(paths),flush=True)
    result=[]
    for i,r in enumerate(pairs):
        ep=ROOT/'datasets/e_ophtha'/r['external_subset'].lower()/f"e_optha_{r['external_subset']}"/r['external_image']
        m,n,ratio,label=compare(cache[str(ep)],cache[str(ROOT/r['train_image'])])
        result.append(dict(**r,orb_matches=m,affine_inliers=n,inlier_ratio=ratio,screen_status=label))
        if (i+1)%1000==0: print('pairs',i+1,flush=True)
    with (OUT/'second_stage_all_pairs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]),lineterminator='\n');w.writeheader();w.writerows(result)
    shortlist=[r for r in result if r['screen_status']!='weak']
    shortlist.sort(key=lambda r:(-r['affine_inliers'],-r['inlier_ratio']))
    with (OUT/'second_stage_review.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(result[0]),lineterminator='\n');w.writeheader();w.writerows(shortlist)
    print('second-stage',dict(Counter(r['screen_status'] for r in result)),flush=True)

if __name__=='__main__': main()
