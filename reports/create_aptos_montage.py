import cv2
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path("datasets/aptos/train.csv")
image_dir = Path("datasets/aptos/train_images")

df = pd.read_csv(csv_path)

fig, axes = plt.subplots(5, 3, figsize=(12, 18))

grade_names = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

for grade in range(5):
    samples = df[df["diagnosis"] == grade].sample(
        n=3, random_state=42
    )

    for col, (_, row) in enumerate(samples.iterrows()):
        image_path = image_dir / f"{row['id_code']}.png"

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ax = axes[grade, col]
        ax.imshow(image)
        ax.set_title(
            f"{grade_names[grade]}\n{row['id_code']}",
            fontsize=9
        )
        ax.axis("off")

plt.suptitle(
    "APTOS 2019 - Visual Inspection by DR Grade",
    fontsize=16
)

plt.tight_layout()

output = Path("reports/aptos_sample_montage.png")
plt.savefig(output, dpi=150, bbox_inches="tight")

print("Montage saved:", output)
print("Images displayed:", 15)
