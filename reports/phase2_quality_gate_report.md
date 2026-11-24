# RetinaSaarthi — Phase 2: Fundus Image Quality Gate Evidence Report

**Project:** Explainable AI for Diabetic Retinopathy Screening in Rural India
**Status:** Prototype evidence review complete; calibration and clinical validation pending
**Report date:** 10 November 2026

## 1. Phase 2 objective

Phase 2 investigates a screening-time quality gate for uploaded fundus photographs. Its purpose is to identify images that are technically unsuitable or uncertain for automated screening and decide whether to:

- **PASS:** allow the image to proceed to subsequent screening stages.
- **RECAPTURE:** request another photograph when a clear technical failure makes the current image unsuitable.
- **MANUAL_REVIEW:** defer the decision when quality or image interpretation is uncertain.

The gate is not a diabetic-retinopathy grading model. Disease severity, lesions, and pathology must not be treated as image-quality defects. The gate must not automatically reject an image merely because it appears to contain more severe disease.

## 2. Dataset context

The project uses three fundus-image datasets:

| Dataset | Images / records | Relevant context |
|---|---:|---|
| APTOS | 3,662 | Five DR grades; labels verified |
| IDRiD segmentation | 81 original images | 54 training and 27 test images; 363 lesion/optic-disc masks |
| IDRiD disease grading | 516 | 413 training and 103 test images |
| Messidor-2 | 1,748 images | 874 examinations; third-party adjudicated grades available for 1,744 images, with 4 ungradable |

Messidor-2's official release does not provide the grading labels used in this analysis. The third-party adjudicated grades are supplementary labels and should be described as such.

These datasets differ in acquisition characteristics, resolution, and label availability. Their quality measurements should not be assumed to be directly interchangeable without further calibration.

## 3. Quality metrics and confounding

The investigation considered image sharpness, illumination, contrast, retinal-field coverage, edge contact, framing, aspect ratio, and mask-derived measurements.

### 3.1 Sharpness and DR-grade confounding

APTOS raw sharpness correlations:

| Comparison | Images | Spearman correlation |
|---|---:|---:|
| DR grade vs raw sharpness | 3,662 | −0.625248 |
| Pixel count vs raw sharpness | 3,662 | −0.719380 |

These strong associations show that raw sharpness is not independent of image resolution and dataset composition.

Resolution-stratified correlations also varied across groups. For example, the 819 × 614 group had a sharpness-versus-grade correlation of approximately −0.180, while the 2416 × 1736 group had a correlation of approximately +0.084. Some groups were small or heavily imbalanced.

**Decision:** retain sharpness as a supporting technical signal, but do not use raw sharpness alone as a rejection criterion. Standardize image dimensions and investigate resolution-aware calibration.

### 3.2 Coverage and mask reliability

The field-coverage measurements require careful interpretation. Mask area divided by its bounding-box area is not the same as the proportion of the full image occupied by the retinal field. Frame coverage should be measured and interpreted separately.

Otsu threshold comparison on 200 images showed substantial variation in mask agreement. Some IoU values were very low, including approximately 0.020 and 0.343, while other images showed much higher agreement.

Synthetic degradation experiments also exposed instability:

- Under occlusion, re-estimated mask coverage decreased strictly across degradation levels in all 200 tested images.
- Under reduced contrast, this strict decrease occurred in only 89 of 200 images.
- Mean re-estimated coverage under occlusion decreased from approximately 71.20% to 61.85% and then 51.58%.
- Reduced-contrast mask estimates were less consistent.

A failed or unstable mask estimate should therefore not automatically trigger RECAPTURE. It should trigger MANUAL_REVIEW when the retinal field cannot be assessed confidently.

### 3.3 Contrast measurement

Contrast measured over different retinal regions can change substantially. In the eroded-mask experiment, mean contrast changes relative to the original were:

| Degradation level | Original contrast | Eroded-region contrast | Change |
|---|---:|---:|---:|
| 1 | 52.005 | 54.280 | +2.275 |
| 2 | 52.005 | 33.422 | −18.583 |
| 3 | 52.005 | 21.810 | −30.195 |

This supports measuring contrast over a consistently defined retinal-interior region rather than relying on an arbitrary mask boundary.

## 4. Synthetic degradation experiments

Synthetic blur, darkening, reduced contrast, and occlusion were used to test whether image metrics respond to controlled technical changes.

The experiments showed:

- Increasing blur reduced sharpness in the tested images.
- Increasing darkening reduced mean illumination and contrast.
- Reduced contrast produced a consistent decrease in the contrast metric in the reported experiment.
- Occlusion affected multiple measurements, but sharpness did not reliably decrease.
- Some metrics changed in directions that were not uniquely attributable to the intended defect.

These experiments demonstrate metric responsiveness under synthetic conditions. They do **not** establish clinical sensitivity, specificity, or real-world performance. Synthetic degradation is not a substitute for naturally occurring poor-quality images or expert-labelled validation data.

## 5. Blinded AI-assisted visual assessment

A blinded review package contained 150 anonymised images:

- 50 APTOS images
- 50 IDRiD images
- 50 Messidor-2 images

Within each dataset, sampling included 30 random and 20 targeted images. Targeted selection was intended to include technically challenging or metric-disagreement cases. The sample is therefore not a population-representative test set.

Gemini and Claude assessed image quality, field completeness, framing, blur, usability, and visible technical defects. The instructions prohibited DR grading and required uncertainty to be recorded rather than treating possible lesions as technical defects.

### 5.1 Reconciliation

**Gemini**
- 150 original assessment rows.
- One duplicate ID was removed, retaining the first occurrence.
- One missing assessment, R-FF36B1DA, was restored from a separately supplied Gemini assessment.
- Final reconciled file: 150 unique IDs matching the review roster.

The restored assessment did not come from the original combined CSV and should be disclosed as a reconciliation caveat.

**Claude**
- 150 original assessment rows.
- 101 verified R-ID assessments mapped to the authoritative roster.
- 49 temporary-ID assessments could not be mapped.
- All 49 remain explicitly marked UNMAPPABLE; no identity was guessed.

The AI-to-AI comparison therefore uses **101 shared images**, not 150.

### 5.2 Categorical agreement

| Assessment field | Exact agreement | Cohen's kappa |
|---|---:|---:|
| Image quality | 69.31% | 0.4911 |
| Field completeness | 74.26% | 0.5146 |
| Field size/framing | 69.31% | 0.4262 |
| Blur at native resolution | 68.32% | 0.3281 |
| Usability decision | 65.35% | 0.3786 |

Image-quality weighted kappa was 0.5965 (linear) and 0.7146 (quadratic). These values indicate stronger agreement when adjacent quality categories receive partial credit.

Important disagreement patterns included:

- Gemini marked 76 images usable, compared with 52 for Claude.
- There were 24 cases where Gemini said usable and Claude said manual review.
- Claude used the uncertain blur category frequently, while Gemini did not use it.
- The reviewers differed substantially on whether the field was large or partially clipped.

Neither reviewer can be considered more accurate based on this comparison. There is no expert reference standard against which to measure correctness.

### 5.3 Defect-tag agreement

Individual binary defect-tag agreement varied:

| Defect | Exact binary agreement | Cohen's kappa |
|---|---:|---:|
| Blur | 89.11% | 0.5346 |
| Clipped | 75.25% | 0.5272 |
| Colour cast | 77.23% | 0.3369 |
| Dark | 94.06% | 0.7884 |
| Glare | 84.16% | 0.5393 |
| Haze | 77.23% | 0.4869 |
| Lesion-heavy appearance | 91.09% | 0.7924 |
| Other | 97.03% | 0.5590 |
| Overexposed | 89.11% | 0.4989 |
| Uneven illumination | 79.21% | 0.5020 |

Despite some high individual-tag agreement, exact agreement on the complete set of defect tags was only 28 of 101 images (27.72%). Agreement on a single defect does not imply agreement on the full defect combination.

The lesion-heavy tag is a descriptive observation, not a technical rejection criterion.

### 5.4 Random versus targeted sample

The shared sample contained 63 random and 38 targeted images.

Exact usability agreement was 68.25% in the random group and 60.53% in the targeted group. Exact complete-tag-set agreement was 31.75% and 21.05%, respectively.

The targeted group had lower exact agreement in four of five categorical fields, although kappa was not uniformly lower. The groups are relatively small and deliberately sampled, so this comparison is exploratory rather than a performance estimate.

### 5.5 Disagreement audit

A disagreement audit found at least one categorical or defect-tag difference in 80 of the 101 shared images (79.2%). Usability had 35 disagreements, blur 32, image quality 31, framing 31, and field completeness 26. Defect-tag sets differed on 76 images.

This reinforces that AI-only assessment is not a dependable ground truth and that disagreements should not be resolved simply by choosing one AI reviewer.

## 6. Manual visual audit of 18 images

A further visual audit covered six images from each dataset. The images were deliberately selected for disagreement analysis, not randomly sampled to estimate population quality.

| Dataset | Usable | Manual review | Recapture |
|---|---:|---:|---:|
| APTOS | 0 | 4 | 2 |
| IDRiD | 3 | 3 | 0 |
| Messidor-2 | 1 | 5 | 0 |
| **Total** | **4** | **12** | **2** |

Recurring observations included:

- Haze and reduced vessel visibility.
- Uneven illumination and locally dark retinal regions.
- Colour casts and washed-out or overexposed areas.
- Bright optic-disc regions and uncertain reflections.
- Partial field clipping, sometimes with central structures still visible.
- Uncertain streaks, pale patches, or other findings that could not safely be classified as technical defects.

Clipping alone was not considered sufficient for recapture when central retinal structures remained visible. Ambiguous reflections or possible retinal findings were not counted as definite technical defects.

**Audit limitation:** the images were inspected using reduced-resolution review sheets. Fine blur at native resolution could not always be determined. These observations are not ophthalmologist or optometrist assessments and are not clinical ground truth.

## 7. Provisional quality-gate design

The evidence supports a conservative three-outcome design.

### PASS

Use when the retinal field is sufficiently visible, the important structures can be assessed, and no clear technical failure prevents the next screening stage.

### RECAPTURE

Use only when a clear technical failure makes the image unsuitable, such as severe blur, severe exposure failure, or substantial obstruction/clipping that prevents meaningful assessment.

### MANUAL_REVIEW

Use when:
- Mask detection fails or is unstable.
- Quality metrics conflict.
- Image quality is borderline or uncertain.
- Illumination, haze, reflections, or colour changes make interpretation uncertain.
- Apparent abnormalities could be pathology rather than acquisition defects.
- Automated measurements cannot confidently establish retinal-field visibility.

No numerical quality thresholds are selected in this phase. Thresholds must not be chosen from AI agreement alone or tuned to DR grade.

## 8. Limitations

1. No ophthalmologist, optometrist, or trained fundus technician supplied a human expert reference assessment.
2. AI-to-AI agreement measures consistency, not correctness or clinical validity.
3. AI reviewers may share biases and are not independent clinical observers.
4. The 18-image visual audit was deliberately selected and is too small to estimate population-level quality or gate performance.
5. Reduced-resolution inspection limits conclusions about native-resolution blur.
6. Synthetic degradations do not fully represent real-world acquisition defects.
7. Mask estimates and contrast measurements are sensitive to segmentation and region selection.
8. Four ungradable Messidor-2 images are insufficient for diagnostic sensitivity or specificity claims.
9. Dataset, camera, resolution, and grading differences may affect metric distributions.
10. No numeric thresholds or clinical operating characteristics have been validated.

## 9. Conclusion and next phase

Phase 2 has completed an exploratory evidence review of image-quality metrics, synthetic degradation responses, mask reliability, blinded AI assessments, and a small manual visual audit.

The evidence supports a **provisional PASS / RECAPTURE / MANUAL_REVIEW architecture**, with MANUAL_REVIEW used whenever the automated evidence is ambiguous or unreliable.

It does not support a claim that the quality gate is clinically validated or that its thresholds are established. The next phase should focus on obtaining expert-labelled quality assessments, defining reproducible defect criteria, calibrating metrics on standardized images, and evaluating the gate on an independent sample.

**Final status:** Phase 2 prototype design and exploratory evidence review complete. Expert review, threshold calibration, independent validation, and clinical evaluation remain pending.

## 10. Supporting evidence files

Key analysis and audit artifacts:

- `reports/quality_raw_sharpness_correlations.csv`
- `reports/quality_stratified_correlations.csv`
- `reports/quality_metric_response_summary.csv`
- `reports/quality_metric_response_summary_v2.csv`
- `reports/quality_mask_iou_otsu_followup2.csv`
- `reports/quality_mask_reestimated_degradation_summary_followup2.csv`
- `reports/quality_contrast_eroded_mask_summary_followup3.csv`
- `reports/quality_review_v3/gemini_assessments_reconciled.csv`
- `reports/quality_review_v3/claude_assessments_mapped.csv`
- `reports/quality_review_v3/claude_assessments_unmappable.csv`
- `reports/quality_review_v3/claude_temp_id_mapping.csv`
- `reports/quality_review_v3/ai_assessment_agreement_report.txt`
- `reports/quality_review_v3/defect_tag_agreement.csv`
- `reports/quality_review_v3/random_vs_targeted_agreement.csv`
- `reports/quality_review_v3/random_vs_targeted_tag_agreement.csv`
- `reports/quality_review_v3/ai_disagreement_visual_audit.csv`
- `reports/quality_review_v3/visual_audit_observations.csv`
- `reports/quality_review_v3/gemini_reconciliation_log.txt`
- `reports/quality_review_v3/claude_reconciliation_log.txt`
