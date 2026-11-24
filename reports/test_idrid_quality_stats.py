import pandas as pd
from scipy.stats import mannwhitneyu

df = pd.read_csv("reports/idrid_quality_metrics.csv")

metrics = [
    "sharpness",
    "brightness",
    "contrast",
    "retinal_coverage_percent"
]

groups = [
    ("All grades", df),
    ("Severe", df[df["true_grade"] == 3]),
    ("PDR", df[df["true_grade"] == 4])
]

results = []

for group_name, subset in groups:
    correct = subset[subset["correct"] == True]
    incorrect = subset[subset["correct"] == False]

    print(f"\n===== {group_name} =====")
    print("Correct:", len(correct))
    print("Incorrect:", len(incorrect))

    for metric in metrics:
        if len(correct) == 0 or len(incorrect) == 0:
            continue

        stat, p = mannwhitneyu(
            correct[metric],
            incorrect[metric],
            alternative="two-sided"
        )

        results.append({
            "group": group_name,
            "metric": metric,
            "correct_median": correct[metric].median(),
            "incorrect_median": incorrect[metric].median(),
            "p_value": p
        })

results_df = pd.DataFrame(results)

# Bonferroni correction across all 12 comparisons
results_df["adjusted_p"] = (
    results_df["p_value"] * len(results_df)
).clip(upper=1.0)

print("\n===== STATISTICAL TEST RESULTS =====")
print(results_df.round(4).to_string(index=False))

print("\nSignificant after correction (p < 0.05):")
significant = results_df[results_df["adjusted_p"] < 0.05]

if significant.empty:
    print("No statistically significant differences.")
else:
    print(significant.to_string(index=False))

results_df.to_csv(
    "reports/idrid_quality_statistical_tests.csv",
    index=False
)

print("\nSaved: reports/idrid_quality_statistical_tests.csv")
