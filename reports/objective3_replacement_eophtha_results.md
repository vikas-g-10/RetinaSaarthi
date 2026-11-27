# Objective 3: replacement model e-ophtha exploratory evaluation

The databases e-ophtha EX, e-ophtha MA and e-ophtha DR are provided free of charge by the partners of the TeleOphta Consortium (see http://cmm.ensmp.fr/Projects/TeleOphta/index.html; [provider and license](https://www.adcis.net/en/third-party/e-ophtha/)). Please also cite Decencière et al., “TeleOphta: Machine learning and image processing methods for teleophthalmology,” IRBM (2013), https://doi.org/10.1016/j.irbm.2013.01.010.

The replacement EfficientNet-B0 was evaluated under the [pre-training protocol](objective3_replacement_eophtha_protocol_v1.md), which retains the earlier fixed e-ophtha design. Both subsets are externally sourced, but this is an **exploratory study**, not independent or clinical validation. The old model's e-ophtha results had already been inspected before replacement training; the duplicate screen's miss rate and patient/eye independence are unverified. The model was selected by APTOS validation macro-F1, not e-ophtha results. Pixel overlap does not establish clinical correctness or the cause of a prediction.

## Checkpoint and provenance

`objective3_replacement_v2/best_model.pth` SHA-256 `91fcd3a6cdb4f238f63affe148a8c87e4ca0685b82e04bd4c4c718e39913ba71` agrees with its saved hash file. Epoch 4 and validation macro-F1 0.6279884534374527 agree with the best of seven history rows; the next three epochs did not improve. Checkpoint provenance and the separate metadata are identical. Saved train and validation manifests, protocol, ImageNet initialization, five code snapshots, image-hash list, decision files and Git patch all match their recorded hashes. The initialization file hash is `7f5810bc96def8f7552d5b7e68d53c4786f81167d28291b21c0d90e1fca14934`. No discrepancy was found in these records. Matching records document this run's inputs; they do not prove all image or patient relationships across sources are absent.

## Methods and results

Predicted-class Grad-CAM at `features[-1]` uses the prior 224×224/ImageNet preprocessing, per-image normalization, bilinear native-resolution mapping and fixed >=0.5 threshold. EX and MA are separate; source-positive mask pixels are provisionally nonzero, and source-negative cases are not given fabricated masks. IoU, Dice and coverage describe overlap with recorded pixels; statistics are unweighted per-image means and medians. The source-negative activated fraction uses the entire image rectangle, including borders. The 20 unique external images in unresolved duplicate comparisons are excluded from primary and reported as sensitivity; shared EX/MA images are included once per applicable analysis and are not independent people.

| Positive cohort | n | Mean / median IoU | Mean / median Dice | Mean / median recorded-pixel coverage |
|---|---:|---:|---:|---:|
| EX primary | 44 | 0.012572 / 0.004297 | 0.024282 / 0.008554 | 0.481445 / 0.539879 |
| MA primary | 143 | 0.000339 / 0.000132 | 0.000677 / 0.000263 | 0.212158 / 0.119576 |
| EX unresolved | 3 | 0.003042 / 0.002553 | 0.006061 / 0.005093 | 0.770311 / 1.000000 |
| MA unresolved | 5 | 0.000661 / 0.000789 | 0.001322 / 0.001578 | 0.542026 / 0.619289 |
| EX combined | 47 | 0.011964 / 0.003086 | 0.023119 / 0.006154 | 0.499884 / 0.540590 |
| MA combined | 148 | 0.000350 / 0.000133 | 0.000699 / 0.000265 | 0.223302 / 0.121692 |

| Source-negative cohort | n | Activated fraction mean / median | Any activation |
|---|---:|---:|---:|
| EX primary | 33 | 0.098679 / 0.096109 | 33/33 |
| MA primary | 221 | 0.099306 / 0.091942 | 221/221 |
| EX unresolved | 2 | 0.129113 / 0.129113 | 2/2 |
| MA unresolved | 12 | 0.101922 / 0.092807 | 12/12 |
| EX combined | 35 | 0.100418 / 0.096702 | 35/35 |
| MA combined | 233 | 0.099440 / 0.091942 | 233/233 |

Source-negative means the source identified the image as negative for that subset's lesion, not necessarily healthy or free of other lesions. Activation in these images is not a validated false-positive detection rate.

## Prespecified sensitivity analyses

| Primary cohort | Threshold | Positive mean IoU / Dice / coverage | Negative mean activated fraction |
|---|---:|---:|---:|
| EX | 0.25 | 0.008082 / 0.015777 / 0.603042 | 0.210941 |
| EX | 0.50 | 0.012572 / 0.024282 / 0.481445 | 0.098679 |
| EX | 0.75 | 0.016235 / 0.031028 / 0.259239 | 0.032703 |
| MA | 0.25 | 0.000319 / 0.000638 / 0.426676 | 0.199587 |
| MA | 0.50 | 0.000339 / 0.000677 / 0.212158 | 0.099306 |
| MA | 0.75 | 0.000367 / 0.000733 / 0.071281 | 0.034895 |

MA 255-only mask sensitivity at 0.5 retains 138/143 primary positives: five masks become empty and are excluded, not relabeled negative. Mean IoU / Dice / coverage are 0.000277 / 0.000553 / 0.217886. Removing only 239 and 247 gives 143 valid cases, with means 0.000339321 / 0.000677406 / 0.212143. Their meanings remain undocumented; sensitivity does not establish annotation semantics. MA marks may be dots or small regions, and spatial resolution affects overlap. Per-image results, all medians and all unresolved/combined sensitivity results are in ignored `outputs/explainability/e_ophtha_replacement_v2/`.

## Verification and reproduction

The replacement-specific evaluator verifies checkpoint, cohort and frozen protocol hashes and fails if its separate output directory already exists. It strictly loads the replacement model state; a focused test compares every loaded tensor with that checkpoint and checks it differs in path from the original. Four focused test files passed 14 tests. An independent arithmetic check reconstructed 876 defined positive-overlap rows and all 42 aggregate groups. All 1,685 scenario rows have the same image hash, mask hash and cohort as the prior evaluation. CPU execution used Python 3.14.10, torch 2.14.1+cu132, torchvision 0.29.1+cu132 and NumPy 2.5.4. The evaluation did not change the prior results or checkpoint.

To reproduce, preserve or move the existing replacement result directory first; the evaluator deliberately will not overwrite it. From the repository root in Fish:

```fish
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python -m pytest -q tests/test_eophtha_replacement.py tests/test_eophtha_evaluation.py tests/test_idrid_overlap.py tests/test_replacement_training.py
env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 .venv/bin/python -m ml.explainability.evaluate_eophtha_replacement
```

Per-image normalization, uncertain annotation completeness, predicted rather than reference grades, arbitrary fixed thresholds, whole-rectangle denominators, the limited duplicate screen, shared visits/images and unknown patient/eye links constrain interpretation. Findings cannot be generalized as clinical performance or independent validation.
