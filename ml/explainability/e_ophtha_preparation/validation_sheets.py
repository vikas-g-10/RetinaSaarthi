import csv
from pathlib import Path
from PIL import Image,ImageDraw
root=Path.cwd(); out=root/'outputs/classification/replacement_eophtha_v1'
rows=list(csv.DictReader((out/'validation_review.csv').open()))
for i,r in enumerate(rows,1):
 a=root/'datasets/e_ophtha'/r['external_subset'].lower()/f"e_optha_{r['external_subset']}"/r['external_image'];b=root/'datasets/aptos/train_images'/f"{r['validation_id']}.png"
 canvas=Image.new('RGB',(2200,850),'white');draw=ImageDraw.Draw(canvas)
 for j,p in enumerate((a,b)):
  with Image.open(p) as im:
   im=im.convert('RGB');im.thumbnail((1080,790),Image.Resampling.LANCZOS);canvas.paste(im,(j*1100+(1100-im.width)//2,50))
  draw.text((j*1100+20,15),f"pair {i}: {'e-ophtha' if j==0 else 'APTOS validation'} {p.name}",fill='black')
 canvas.save(out/f'validation_pair_{i:02}.png')
print('wrote',len(rows),'sheets')
