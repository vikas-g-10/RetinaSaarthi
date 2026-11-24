
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS = (
    ROOT
    / "outputs/classification/combined_aptos_idrid_baseline"
    / "validation_analysis/validation_predictions.csv"
)

IMAGE_DIR = ROOT / "datasets/aptos/train_images"

OUTPUT = (
    ROOT
    / "reports/severe_dr_validation_contact_sheet.png"
)

df = pd.read_csv(PREDICTIONS)

severe = df[df["true_grade"] == 3].copy()
severe = severe.sort_values(
    ["correct", "image_id"],
    ascending=[False, True],
)

fig, axes = plt.subplots(
    5,
    6,
    figsize=(20, 17),
)

axes = axes.flatten()

for index, (_, row) in enumerate(severe.iterrows()):
    image_path = IMAGE_DIR / f"{row['image_id']}.png"

    image = Image.open(image_path).convert("RGB")

    axes[index].imshow(image)
    axes[index].axis("off")

    status = "CORRECT" if row["correct"] else "WRONG"

    predicted_grade = int(row["predicted_grade"])
    confidence = float(row["confidence"])

    axes[index].set_title(
        f"{row['image_id']}\n"
        f"Predicted: {predicted_grade} | "
        f"Confidence: {confidence:.2f}\n"
        f"{status}",
        fontsize=8,
    )

for index in range(len(severe), len(axes)):
    axes[index].axis("off")

fig.suptitle(
    "Severe DR Validation Images — Original Combined Model\n"
    "Green group first: correct predictions | Remaining: misclassified",
    fontsize=15,
)

plt.tight_layout()

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT, dpi=160, bbox_inches="tight")
plt.close()

print("Total images:", len(severe))
print("Correct:", int(severe["correct"].sum()))
print("Misclassified:", int((~severe["correct"]).sum()))
print("Contact sheet saved:", OUTPUT)
