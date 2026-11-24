from pathlib import Path
import pandas as pd

base = Path("reports")

files = [
    "quality_raw_sharpness_correlations.csv",
    "quality_stratified_correlations.csv",
    "quality_metric_response_summary.csv",
    "quality_metric_response_summary_v2.csv",
    "quality_mask_iou_otsu_followup2.csv",
    "quality_mask_reestimated_degradation_summary_followup2.csv",
    "quality_contrast_eroded_mask_summary_followup3.csv",
    "quality_review_v3/defect_tag_agreement.csv",
    "quality_review_v3/random_vs_targeted_agreement.csv",
    "quality_review_v3/random_vs_targeted_tag_agreement.csv",
]

for filename in files:
    path = base / filename

    print("\n" + "=" * 75)
    print("FILE:", path)
    print("=" * 75)

    if not path.exists():
        print("MISSING")
        continue

    try:
        data = pd.read_csv(path)
        print("Shape:", data.shape)
        print("Columns:", list(data.columns))
        print(data.head(12).to_string(index=False))
    except Exception as error:
        print("READ ERROR:", error)

text_files = [
    "quality_review_v3/ai_assessment_agreement_report.txt",
    "quality_review_v3/gemini_reconciliation_log.txt",
    "quality_review_v3/claude_reconciliation_log.txt",
]

for filename in text_files:
    path = base / filename

    print("\n" + "=" * 75)
    print("TEXT FILE:", path)
    print("=" * 75)

    if path.exists():
        print(path.read_text())
    else:
        print("MISSING")
