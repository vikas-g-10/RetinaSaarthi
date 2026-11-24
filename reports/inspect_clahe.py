
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw

from ml.classification.transforms_clahe import ApplyCLAHE


ROOT = Path(__file__).resolve().parents[1]

manifest = pd.read_csv(
    ROOT / "outputs/classification/splits/combined_train_aptos_idrid.csv"
)

settings = [
    ("Original", None),
    ("CLAHE 1.0", 1.0),
    ("CLAHE 1.5", 1.5),
    ("CLAHE 2.0", 2.0),
]

cell = 448
header = 35
row_height = cell + header

canvas = Image.new(
    "RGB",
    (cell * len(settings), row_height * 10),
    "white",
)

draw = ImageDraw.Draw(canvas)

row_number = 0

for source in ["APTOS", "IDRiD"]:
    for grade in range(5):
        subset = manifest[
            (manifest["source"] == source)
            & (manifest["diagnosis"] == grade)
        ].sort_values("id_code")

        record = subset.iloc[0]
        image_path = ROOT / record["image_path"]

        with Image.open(image_path) as img:
            original = img.convert("RGB")

        y = row_number * row_height

        for column, (label, clip_limit) in enumerate(settings):
            if clip_limit is None:
                result = original.copy()
            else:
                clahe = ApplyCLAHE(
                    clip_limit=clip_limit,
                    tile_grid_size=(8, 8),
                )
                result = clahe(original)

            result = result.resize((cell, cell))

            x = column * cell

            draw.text(
                (x + 10, y + 10),
                label,
                fill="black",
            )

            canvas.paste(result, (x, y + header))

        print(f"Processed {source}, grade {grade}: {record['id_code']}")
        row_number += 1

output = ROOT / "reports/clahe_strength_comparison.png"

canvas.crop(
    (0, 0, cell * len(settings), row_number * row_height)
).save(output)

print(f"\nSaved comparison: {output}")
