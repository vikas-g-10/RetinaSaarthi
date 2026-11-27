"""Produce contact sheets for all second-stage review pairs."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/explainability/e_ophtha_review'
rows=list(csv.DictReader((OUT/'second_stage_review.csv').open()))
for start in range(0,len(rows),4):
    group=rows[start:start+4]
    fig,axes=plt.subplots(len(group),2,figsize=(13,4*len(group)),squeeze=False)
    fig.subplots_adjust(top=.94,bottom=.03,hspace=.3)
    for k,(axrow,r) in enumerate(zip(axes,group),start+1):
        paths=[ROOT/'datasets/e_ophtha'/r['external_subset'].lower()/f"e_optha_{r['external_subset']}"/r['external_image'],ROOT/r['train_image']]
        for ax,p,label in zip(axrow,paths,['e-ophtha','current-manifest image']):
            with Image.open(p) as im:
                im.thumbnail((850,550)); ax.imshow(im)
            ax.set_title(f'{k}: {label} | {p.name}',fontsize=10); ax.axis('off')
        axrow[0].text(.03,-.07,f"ORB inliers={r['affine_inliers']}/{r['orb_matches']} | fraction={float(r['inlier_ratio']):.2f}",transform=axrow[0].transAxes,fontsize=10)
    fig.suptitle('Second-stage candidate review: anatomical correspondence is unverified',fontsize=14)
    fig.savefig(OUT/f'second_stage_sheet_{start//4+1:02d}.png',dpi=130);plt.close(fig)
print('sheets', (len(rows)+3)//4,'pairs',len(rows))
