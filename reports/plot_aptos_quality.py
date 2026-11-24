import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("reports/aptos_quality_metrics.csv")

metrics = [
    "sharpness",
    "brightness",
    "contrast",
    "retinal_coverage_percent"
]

titles = [
    "Sharpness Distribution",
    "Brightness Distribution",
    "Contrast Distribution",
    "Retinal Coverage Distribution"
]

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

for ax, metric, title in zip(axes.flat, metrics, titles):
    ax.hist(df[metric], bins=40, edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(metric)
    ax.set_ylabel("Number of images")
    ax.grid(alpha=0.2)

plt.tight_layout()

output = Path("reports/aptos_quality_distributions.png")
plt.savefig(output, dpi=150, bbox_inches="tight")

print("Distribution plots saved:", output)

# Compare sharpness across DR grades
plt.figure(figsize=(10, 6))

df.boxplot(column="sharpness", by="diagnosis")

plt.title("Sharpness Distribution by DR Grade")
plt.suptitle("")
plt.xlabel("DR Grade")
plt.ylabel("Laplacian Variance")
plt.grid(alpha=0.2)

output = Path("reports/aptos_sharpness_by_grade.png")
plt.savefig(output, dpi=150, bbox_inches="tight")

print("Grade comparison saved:", output)
