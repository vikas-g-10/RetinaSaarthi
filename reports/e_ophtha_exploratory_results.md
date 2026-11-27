# e-ophtha external exploratory Grad-CAM study

The databases e-ophtha EX, e-ophtha MA and e-ophtha DR are provided free of charge by the partners of the TeleOphta Consortium (see http://cmm.ensmp.fr/Projects/TeleOphta/index.html; [provider and license](https://www.adcis.net/en/third-party/e-ophtha/)). Please also cite Decencière et al., “TeleOphta: Machine learning and image processing methods for teleophthalmology,” IRBM (2013), https://doi.org/10.1016/j.irbm.2013.01.010.

The separate lesion-specific analyses are complete under the pre-inference [version 2 protocol](e_ophtha_protocol_v2.md). This is externally sourced exploratory work, not independent or clinical validation. The checkpoint's historical training manifest is unknown, image- and patient/eye-level independence are unverified, and the prior duplicate gate did not establish real-world miss rate. Model outputs did not inform cohort decisions.

The 22 prioritized pairs received individual vessel/optic-disc comparison and were all classified unresolved; no confirmed or plausible same-image match was assigned. The affected 20 distinct images are excluded from primary and retained in a separate sensitivity cohort. Because shared EX/MA images are removed from each applicable subset, the unresolved group has 22 subset records, not 22 independent images. The remaining 6,980 weak pairs and first-stage non-candidates are not proven independent. The review was non-expert and cannot rule out transformations or another photograph of the same eye. See the ignored `outputs/explainability/e_ophtha_review/decisions_v2.csv`, `detail_pair_01.png`–`detail_pair_22.png`, and screening records for individual rationale and visual evidence.

The source-positive primary analysis uses predicted-class Grad-CAM, `features[-1]`, the existing 224×224/ImageNet input and a fixed 0.5 cutoff after bilinear mapping to native coordinates. Recorded masks use nonzero pixels; mask meanings for unusual MA values are undocumented. Means and medians weight images equally. Coverage is of recorded pixels only, not lesion-instance detection; unannotated pixels cannot be assumed lesion-free. Source-negative means negative for the relevant dataset lesion category only. Per-image results and full sensitivity summaries are in ignored `outputs/explainability/e_ophtha_evaluation_v2/per_image.csv` and `aggregate.csv`.

| Subset / cohort | Source-positive n | IoU mean / median | Dice mean / median | Coverage mean / median |
|---|---:|---:|---:|---:|
| EX primary | 44 | 0.010334 / 0.002403 | 0.020005 / 0.004795 | 0.434039 / 0.361719 |
| MA primary | 143 | 0.000356 / 0.000124 | 0.000712 / 0.000248 | 0.203216 / 0.118200 |
| EX unresolved sensitivity | 3 | 0.004545 / 0.004682 | 0.009043 / 0.009321 | 0.937309 / 1.000000 |
| MA unresolved sensitivity | 5 | 0.000601 / 0.000513 | 0.001200 / 0.001025 | 0.456595 / 0.570755 |
| EX combined sensitivity | 47 | 0.009965 / 0.002604 | 0.019306 / 0.005194 | 0.466163 / 0.456606 |
| MA combined sensitivity | 148 | 0.000365 / 0.000133 | 0.000728 / 0.000267 | 0.211776 / 0.128598 |

| Subset / cohort | Source-negative n | Activated whole-image fraction mean / median | Images with any activation |
|---|---:|---:|---:|
| EX primary | 33 | 0.099992 / 0.087695 | 33/33 |
| MA primary | 221 | 0.093972 / 0.085918 | 221/221 |
| EX unresolved sensitivity | 2 | 0.090989 / 0.090989 | 2/2 |
| MA unresolved sensitivity | 12 | 0.087031 / 0.083685 | 12/12 |
| EX combined sensitivity | 35 | 0.099477 / 0.087695 | 35/35 |
| MA combined sensitivity | 233 | 0.093615 / 0.085399 | 233/233 |

The negative denominator is the entire image rectangle, including black borders, and varies with framing; this is not a false-positive lesion detection rate. Maps are normalized separately per image and generated for the predicted grade, which is not a known reference grade for this dataset. Apparent coverage can therefore coexist with a broad highlighted region, and the small MA dots make pixel overlap and boundaries especially resolution-sensitive.

## Prespecified sensitivities

| Primary cohort, nonzero mask | Cutoff | Positive mean IoU / Dice / coverage | Negative mean activated fraction |
|---|---:|---:|---:|
| EX | 0.25 | 0.007267 / 0.014172 / 0.592478 | 0.204604 |
| EX | 0.50 | 0.010334 / 0.020005 / 0.434039 | 0.099992 |
| EX | 0.75 | 0.012412 / 0.023731 / 0.207163 | 0.031792 |
| MA | 0.25 | 0.000321 / 0.000642 / 0.412339 | 0.188498 |
| MA | 0.50 | 0.000356 / 0.000712 / 0.203216 | 0.093972 |
| MA | 0.75 | 0.000259 / 0.000516 / 0.053598 | 0.031808 |

For MA at 0.5, the 255-only mask sensitivity retains 138 of 143 primary positives; five become empty and are excluded from overlap, never relabeled negative. Mean IoU / Dice / coverage are 0.000305 / 0.000608 / 0.213729. Excluding just 239 and 247 gives 143 valid cases and the same displayed primary means (0.000356 / 0.000712 / 0.203216). The full CSV contains per-image values, medians, all combined and unresolved sensitivity groups, and the five affected identities. The cutoff was fixed rather than selected using these results. The sensitivity findings do not resolve the undocumented mask-value semantics.

## Reproduction and constraints

From the repository root in Fish, using the preserved local inventory, review outputs, source datasets and checkpoint:

```fish
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_eophtha_evaluation.py tests/test_idrid_overlap.py
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python -m ml.explainability.evaluate_eophtha
```

The evaluation refuses changed protocol, cohort and checkpoint hashes, checks image hashes and mask dimensions/values, records mask hashes per image, and writes only to the ignored `outputs/explainability/e_ophtha_evaluation_v2/` directory. `ml/explainability/create_eophtha_cohort_v2.py` documents cohort derivation from the preserved screening shortlist and inventory; do not regenerate after changing decisions without versioning the protocol. A repeat CPU run produced an identical aggregate CSV SHA-256 (`6c000f2ef179b4b67b7da7aff253bdc60d23e019530c4df3a06cae2acbb0f505`). An independent arithmetic check reconstructed all 42 aggregate groups and 876 defined positive overlap rows from the per-image CSV. Metadata records Python, package versions and device; small floating-point differences across environments are possible. The run used CPU, Python 3.14.10, torch 2.14.1+cu132, torchvision 0.29.1+cu132 and NumPy 2.5.4. The 29 shared EX/MA images contribute to both applicable analyses but are not independent subjects. Distinct visit folders are not verified distinct patients. No clinical conclusions or patient-level confidence intervals follow from these results.
