# RetinaSaarthi Phase 1 Summary

Phase 1 covers dataset collection, label inspection, image-quality analysis, IDRiD correspondence, duplicate review, and segmentation-mask eligibility. This report records only observed files and completed outputs. It does not establish clinical validity.

## Datasets and counts

### APTOS 2019

APTOS contains 3,662 training images. The label distribution is:

| DR class | Images |
|---:|---:|
| 0 | 1,805 |
| 1 | 370 |
| 2 | 999 |
| 3 | 193 |
| 4 | 295 |

The distribution is imbalanced, with class 0 the largest class and classes 1 and 3 substantially smaller than classes 0 and 2. **Source:** [`datasets/aptos/train.csv`](../datasets/aptos/train.csv) has 3,663 lines including its header; the image directory contains 3,662 PNG files.

### IDRiD segmentation

The segmentation set contains 81 original images: 54 training images and 27 testing images. The segmentation ground truths are organized into Microaneurysms (MA), Haemorrhages (HE), Hard Exudates (EX), Soft Exudates (SE), and Optic Disc (OD) folders. **Source:** segmentation original-image directories under [`datasets/idrid/segmentation/`](../datasets/idrid/segmentation/); the 81-row correspondence input is [`reports/idrid_mapping/mapping.csv`](idrid_mapping/mapping.csv).

### IDRiD disease grading

The grading set contains 516 original images: 413 training images and 103 testing images. **Source:** grading original-image directories under [`datasets/idrid/disease_grading/`](../datasets/idrid/disease_grading/); the label CSVs are under [`datasets/idrid/disease_grading/`](../datasets/idrid/disease_grading/).

### Messidor-2
Messidor-2 image filenames and third-party adjudicated grades were verified from [`messidor_data.csv`](../datasets/messidor2/labels/messidor_data.csv), which is separate from the official Messidor-2 release. The official ADCIS release provides the image-pair table but no grades. The adjudicated source contains `1,748` rows: `1,744` carry a DR grade in `{0,1,2,3,4}` and `4` are ungradable. The DR-grade distribution is `0: 1,017; 1: 270; 2: 347; 3: 75; 4: 35`. The DME distribution is `0: 1,593; 1: 151; blank: 4`, and the gradability distribution is `gradable=1: 1,744; gradable=0: 4`. PNG/Original has `1,058` images and JPEG/Extension has `690`; these agree with the expected counts. The image-pair table contains `874` exams, corresponding to `529` Original exams and `345` Extension exams. **Source:** [`reports/messidor2_labels_verified.csv`](messidor2_labels_verified.csv), [`datasets/messidor2/labels/messidor_data.csv`](../datasets/messidor2/labels/messidor_data.csv), [`datasets/messidor2/messidor-2.csv`](../datasets/messidor2/messidor-2.csv), and [`reports/messidor2_labels_integration_output.txt`](messidor2_labels_integration_output.txt).

The native adjudicated grade classes were preserved without remapping or merging. The verified distributions, gradability counts, eye-pair checks, resolution/file-format tables, and association statistics are recorded in the script output and derived CSV. These are dataset-association checks, not clinical validation. **Source:** [`reports/messidor2_labels_verified.csv`](messidor2_labels_verified.csv) and [`reports/integrate_messidor2_labels.py`](integrate_messidor2_labels.py).

The two eyes of an exam have different adjudicated DR grades in `207` of `870` fully graded exams. Confidence intervals for exam-level Messidor-2 results must resample exams, not images, because the two eyes are paired within exams. **Source:** [`reports/messidor2_labels_integration_output.txt`](messidor2_labels_integration_output.txt).

The resolution-grade Spearman correlation is `-0.25599900780173784` (approximately `-0.256`) in Messidor-2 versus `+0.6073177092327471` in APTOS: the signs are opposite, so Messidor-2 is a stress test for resolution shortcuts. Grade distribution also differs strongly between the PNG (Original) and JPG (Extension) subsets, so results should report both subsets separately as well as combined. For the gradable-only chi-square tables, Cramér's V is `0.247478` for resolution and `0.365471` for file format, using `V = sqrt(chi-square / (n * min(r-1, c-1)))` with `n = 1,744` gradable images; the resolution table has `r=3` and `c=5`, and the format table has `r=2` and `c=5`. **Source:** [`reports/messidor2_labels_integration_output.txt`](messidor2_labels_integration_output.txt), [`reports/aptos_quality_metrics.csv`](aptos_quality_metrics.csv), and [`reports/analyze_aptos_quality.py`](analyze_aptos_quality.py).

The four ungradable filenames—`20060411_58550_0200_PP.png`, `IM002385.jpg`, `IM003718.jpg`, and `IM004176.jpg`—could serve as a possible tiny independent check for a Phase 2 quality gate, but `n=4` cannot validate that gate. **Source:** [`reports/messidor2_labels_integration_output.txt`](messidor2_labels_integration_output.txt).

Use and acknowledgement requirements for this third-party grade source: research and educational use; no redistribution; acknowledge the LaTIM laboratory and Messidor programme partners; cite Decenciere et al. (2014) and Abramoff et al. (2013), with Krause et al. (2018) for the adjudicated grades. **Source:** [ADCIS Messidor-2 page](https://www.adcis.net/en/third-party/messidor2/) for licence terms, exam counts, and the absence of official annotations; Kaggle [`google-brain/messidor2-dr-grades`](https://www.kaggle.com/datasets/google-brain/messidor2-dr-grades) for adjudicated grades and Krause et al. (2018).

## APTOS image-quality findings

The quality report contains one row per image and records width, height, sharpness, brightness, contrast, and retinal coverage. The quality analysis uses an analysis-only non-black-pixel mask; no quality filtering threshold was decided.

Mean resolution and sharpness by DR class were:

| DR class | Images | Mean width | Mean height | Mean pixels | Mean sharpness |
|---:|---:|---:|---:|---:|---:|
| 0 | 1,805 | 1,433.416 | 1,206.675 | 1,977,060.939 | 45.891 |
| 1 | 370 | 2,471.446 | 1,784.665 | 4,651,901.584 | 14.177 |
| 2 | 999 | 2,603.877 | 1,846.462 | 5,158,814.917 | 16.099 |
| 3 | 193 | 2,590.746 | 1,840.819 | 5,000,506.653 | 15.277 |
| 4 | 295 | 2,632.339 | 1,874.522 | 5,338,439.864 | 15.719 |

Resolution is associated with grade in this dataset: the computed Spearman correlation between diagnosis and total pixels is `0.6073177092327471`. Class 0 images are, on average, much smaller than classes 1–4, and class 0 also has a higher mean sharpness (`45.891`) than the other classes shown above. This is a possible shortcut: a model could use acquisition/resolution or related image characteristics rather than retinal pathology. These observations do **not** define quality thresholds or a quality-based exclusion rule. **Source:** [`reports/aptos_quality_metrics.csv`](aptos_quality_metrics.csv), analyzed by [`reports/analyze_aptos_quality.py`](analyze_aptos_quality.py).

## IDRiD correspondence

The file numbers do **not** link the segmentation and grading sets. In the all-vs-all experiment, all 54 segmentation queries had their same-numbered grading image ranked below the top match: top-1 was `0/54`, and the median true-partner rank was `183.0`. With 413 grading candidates, the random rank expectation is `207`. **Source:** [`reports/idrid_allvsall_train/allvsall_results.csv`](idrid_allvsall_train/allvsall_results.csv), [`reports/match_idrid_allvsall.py`](match_idrid_allvsall.py), and the grading image count above.

Content-based matching then found 58 confident pairs among the 81 segmentation images: 45 segmentation-train pairs and 13 segmentation-test pairs. For those confident rows, mean pixel MAE was `0.6582758620689655`, with MAE ranging from `0.0` to `1.35`; pixel SSIM ranged from `0.9428` to `1.0`. These pairs are pixel-aligned, so no registration is needed. There are 16 confident pairs crossing the train/test split. **Source:** [`reports/idrid_mapping/mapping.csv`](idrid_mapping/mapping.csv) and its computed confident-row metrics.

ORB, SIFT, and ECC registration experiments were explored in the retained report outputs, but direct content matching and pixel checks already established the usable aligned pairs. Registration was therefore unnecessary for the final correspondence decisions; the experiments should not be described as failures of the IDRiD data. **Source:** [`reports/register_idrid_pairs.py`](register_idrid_pairs.py), [`reports/register_idrid_sift.py`](register_idrid_sift.py), [`reports/register_idrid_ecc.py`](register_idrid_ecc.py), and the corresponding `reports/idrid_registration*/` outputs.

### Corrections

The earlier mean SSIM `0.7281` for same-numbered segmentation/grading pairs was measured on mismatched pairs. It must not be cited as evidence of correspondence. The corrected same-numbered comparison output has mean SSIM `0.7281087189612623`; its mean MAE is `18.9836457382716`, consistent with the fact that those same-numbered pairs are not the established content matches. **Source:** [`reports/idrid_image_similarity_corrected.csv`](idrid_image_similarity_corrected.csv), [`reports/idrid_image_similarity.csv`](idrid_image_similarity.csv), and [`reports/compare_idrid_images.py`](compare_idrid_images.py).

## IDRiD grading duplicates and eligibility

The stronger grading-set duplicate audit found six verified duplicate pairs. Four pairs have label disagreement and two pairs agree:

| Pair | Labels | Agreement |
|---|---|---|
| train `IDRiD_021.jpg` / train `IDRiD_391.jpg` | `(DR 1, ME 0)` / `(DR 2, ME 0)` | disagree |
| train `IDRiD_028.jpg` / train `IDRiD_327.jpg` | `(DR 4, ME 2)` / `(DR 3, ME 2)` | disagree |
| train `IDRiD_045.jpg` / train `IDRiD_333.jpg` | `(DR 2, ME 0)` / `(DR 2, ME 1)` | disagree |
| train `IDRiD_073.jpg` / train `IDRiD_375.jpg` | `(DR 2, ME 1)` / `(DR 2, ME 1)` | agree |
| train `IDRiD_079.jpg` / train `IDRiD_279.jpg` | `(DR 1, ME 0)` / `(DR 1, ME 0)` | agree |
| test `IDRiD_064.jpg` / train `IDRiD_118.jpg` | `(DR 3, ME 0)` / `(DR 0, ME 0)` | disagree |

The duplicate report uses a strict clique rule: every pair inside a verified group passes the correlation and pixel checks. **Source:** [`reports/idrid_grading_duplicates.csv`](idrid_grading_duplicates.csv) and [`reports/analyze_idrid_grading_duplicates.py`](analyze_idrid_grading_duplicates.py).

Test `IDRiD_064.jpg` duplicates train `IDRiD_118.jpg` with conflicting DR grades (`3` versus `0`); this is the one duplicate pair crossing the train/test split, and four of the six duplicate pairs have conflicting labels. **Source:** [`reports/idrid_grading_duplicates.csv`](idrid_grading_duplicates.csv).

The six held-out imperfect segmentation images are train `IDRiD_39.jpg`, `IDRiD_26.jpg`, `IDRiD_11.jpg`, and `IDRiD_25.jpg`, plus test `IDRiD_56.jpg` and `IDRiD_70.jpg`. The eligibility report contains 15 unpaired segmentation images; these are train `IDRiD_04.jpg`, `IDRiD_12.jpg`, `IDRiD_30.jpg`, `IDRiD_32.jpg`, and `IDRiD_45.jpg`, plus test `IDRiD_55.jpg`, `IDRiD_57.jpg`, `IDRiD_60.jpg`, `IDRiD_61.jpg`, `IDRiD_64.jpg`, `IDRiD_65.jpg`, `IDRiD_66.jpg`, `IDRiD_68.jpg`, `IDRiD_75.jpg`, and `IDRiD_80.jpg`. **Source:** [`reports/idrid_eligibility.csv`](idrid_eligibility.csv) and [`reports/idrid_mapping/mapping.csv`](idrid_mapping/mapping.csv).

The final eligibility table has one row for each of the 81 segmentation images. Its primary eligible set is the 59 images with status `confident` or `tie_duplicate`; the tie-label-conflict image is excluded from that primary set. Mask-file-present counts are:

Among the 59 primary eligible images, the DR-grade distribution is `2: 37`, `3: 14`, and `4: 8`; no grade `0` or `1` image is eligible, so explanation validation does not cover healthy or mild disease. Segmentation test `IDRiD_62.jpg` is `tie_duplicate`: grading train `IDRiD_073.jpg = IDRiD_375.jpg`, with agreeing labels. Segmentation test `IDRiD_69.jpg` is `tie_label_conflict`: grading train `IDRiD_327.jpg = IDRiD_028.jpg`, with DR `3` versus `4`; it is excluded from the primary set. **Source:** [`reports/idrid_eligibility.csv`](idrid_eligibility.csv).

| Mask file present | Excluding tie-label conflict | Including tie-label conflict |
|---|---:|---:|
| MA | 59 | 60 |
| HE | 58 | 59 |
| EX | 59 | 60 |
| SE | 28 | 28 |
| OD | 59 | 60 |

`has_MA`, `has_HE`, `has_EX`, `has_SE`, and `has_OD` mean that a mask file is present. A false value does not mean that a lesion is absent. OD is an anatomical structure, not a diabetic-retinopathy lesion. **Source:** [`reports/idrid_eligibility.csv`](idrid_eligibility.csv), generated by [`reports/build_idrid_eligibility.py`](build_idrid_eligibility.py).

## Limitations and decisions for later phases

- APTOS is the primary training set. IDRiD and Messidor-2 are external datasets for correspondence checks, segmentation evidence, label analysis, and later external evaluation.
- Messidor-2 images and adjudicated grades were integrated and verified. **Source:** [`reports/messidor2_labels_integration_output.txt`](messidor2_labels_integration_output.txt).
- Grad-CAM is an explanation/support signal, not proof of clinical correctness or clinical validation.
- IoU and Dice are overlap measures only; they do not establish clinical correctness.
- For validation, Grad-CAM should be computed on the segmentation images themselves so the heatmap and the corresponding lesion mask share a coordinate frame.
- No APTOS image-quality threshold or exclusion policy was decided in Phase 1.
- The IDRiD duplicate report does not decide which duplicate to drop; it records duplicate evidence and label agreement/disagreement.
- Soft-exudate masks have only `28` eligible images, so per-lesion results must be reported separately. **Source:** [`reports/idrid_eligibility.csv`](idrid_eligibility.csv).
- Microaneurysm masks are much smaller than a coarse Grad-CAM cell, so overlap metrics alone are a poor measure for them; a complementary measure is needed in Phase 4.
- Label conflicts among duplicate IDRiD grading images limit the reliability of any IDRiD accuracy figure.
- The eligible set contains only grades `2`–`4`. **Source:** [`reports/idrid_eligibility.csv`](idrid_eligibility.csv).

## Phase 1 checklist

Complete:

- APTOS labels, class imbalance, and image-quality metrics inspected.
- IDRiD segmentation and grading inventories counted.
- IDRiD numbering mismatch established with all-vs-all matching.
- 58 confident content-based IDRiD pairs recorded; 16 are cross-split. **Source:** [`reports/idrid_mapping/mapping.csv`](idrid_mapping/mapping.csv).
- Registration was shown to be unnecessary for the accepted aligned pairs.
- Grading duplicates and label conflicts audited.
- Segmentation mask filename mapping validated with zero masks lacking a same-split original.
- [`reports/idrid_eligibility.csv`](idrid_eligibility.csv) created with 81 rows, pairing status, grading labels, duplicate-conflict flag, and mask-file-presence columns. **Source:** [`reports/idrid_eligibility.csv`](idrid_eligibility.csv).
- Assemble/extract Messidor-2 and perform its label/image audit. **Source:** [`reports/messidor2_labels_integration_output.txt`](messidor2_labels_integration_output.txt).

Open:

- Decide any APTOS quality handling only after model-validation experiments; no threshold has been selected here.
- Train/evaluate the primary APTOS model.
- Validate explanations and segmentation overlap using the coordinate-aligned IDRiD workflow.
- Treat all later performance and explanation results as screening-support evidence, not clinical validation.
