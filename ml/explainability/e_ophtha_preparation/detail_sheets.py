import csv
from pathlib import Path
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'outputs/explainability/e_ophtha_review'
rows=list(csv.DictReader((OUT/'second_stage_review.csv').open()))
for idx,r in enumerate(rows,1):
    a=ROOT/'datasets/e_ophtha'/r['external_subset'].lower()/f"e_optha_{r['external_subset']}"/r['external_image']
    b=ROOT/r['train_image']
    canvas=Image.new('RGB',(2200,850),'white'); draw=ImageDraw.Draw(canvas)
    for j,p in enumerate((a,b)):
        with Image.open(p) as im:
            im=im.convert('RGB'); im.thumbnail((1080,790),Image.Resampling.LANCZOS)
            canvas.paste(im,(j*1100+(1100-im.width)//2,50))
        draw.text((j*1100+20,15),f"pair {idx}: {'e-ophtha' if j==0 else 'training'} {p.name}",fill='black',stroke_width=0)
    canvas.save(OUT/f'detail_pair_{idx:02d}.png')
print('saved',len(rows),'detail images')
