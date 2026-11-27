# Objective 3: exploratory IDRiD overlap results

Protocol: [objective3_protocol.md](objective3_protocol.md). Run from the repository root with an environment containing the project dependencies:

```fish
.venv/bin/python reports/check_objective3_alignment.py
env CUDA_VISIBLE_DEVICES='' .venv/bin/python -m ml.explainability.evaluate_idrid_overlap --figures
.venv/bin/python -m pytest tests/test_idrid_overlap.py tests/test_objective3_alignment.py -q
```

The last evaluation used CPU and the local Python 3.14 environment. A fresh run requires the local datasets, the specified checkpoint and the project dependencies; another software or hardware environment may yield small numerical differences. The generated per-image CSV and alignment patch evidence are stored under the ignored `outputs/explainability/idrid_overlap/` directory. The alignment CSV records selected component IDs, centers, sizes, skipped boundary cases, offsets, correlations, and input SHA-256 hashes. The fixed normalized activation threshold is 0.5. Coverage is the fraction of recorded lesion pixels within the thresholded activation. IoU and Dice also penalize activation outside the recorded union; missing annotations make that interpretation uncertain.

The `--figures` option also generates one `IDRiD_<seg>__IDRiD_<grade>_review.png` per pair in that directory, using the same arrays as the CSV calculation. It requires Matplotlib, which is used by the existing Grad-CAM script but is not listed in the project's requirements file. Each figure shows the grading image, predicted-class overlay, available mask union, and a color-coded threshold comparison. The figures are downsampled for display: small lesions can be hard to see or disappear, so do not infer exact pixel counts from the image. The heatmap overlay is illustrative; the comparison and metrics use the bilinearly resized continuous map and native-resolution masks. Missing files remain unknown, including SE for 09.

| Segmentation / grading test | Predicted grade | Available masks | IoU | Dice | Lesion coverage |
|---|---:|---|---:|---:|---:|
| 09 / 091 | 2 | MA, HE, EX; SE missing | 0.0902 | 0.1655 | 0.4030 |
| 22 / 088 | 3 | MA, HE, EX, SE | 0.0533 | 0.1012 | 0.3418 |
| 31 / 089 | 4 | MA, HE, EX, SE | 0.0360 | 0.0696 | 0.2063 |
| Mean, four mask files present (n=2) | — | — | 0.0447 | 0.0854 | 0.2741 |

All reference grades are 2. The model's predicted-class maps explain grades 3 and 4 for the last two images, not the reference grade. These numbers describe only this tiny sample and threshold; they do not establish clinical correctness or validation.

## Visual review

In each review, orange marks recorded lesion pixels outside the thresholded region, blue marks highlighted pixels outside the available mask union, and yellow marks their intersection. Blue pixels are not necessarily lesion-free: the available masks may be incomplete. The panels are downsampled and cannot show every small lesion.

- **09 / 091:** The highlighted region crosses part of the central recorded lesion cluster; many recorded pixels lie outside it, particularly around the lower and outer parts of that cluster. Highlighted pixels also extend beyond the available masks. The SE mask file is missing and its contents are unknown.
- **22 / 088:** The highlighted region covers some recorded lesions in the upper-left and central areas of the comparison panel. Other recorded lesions, including areas on the right and lower parts, lie outside the region; substantial highlighted area lies outside the available masks. The predicted grade is 3, while the reference is 2.
- **31 / 089:** The highlighted region intersects a central recorded lesion area and some nearby small regions; much of the recorded union, including upper and right-side areas, is outside it. Much of the highlighted area is outside the available masks. The predicted grade is 4, while the reference is 2.

These observations describe spatial overlap at the fixed threshold; they do not explain the causes of the model's predictions or errors.

## Larger evaluation feasibility

The reproducible [candidate inventory](objective3_candidate_inventory.csv) covers all 81 segmentation images. Run `.venv/bin/python reports/build_objective3_inventory.py` from the repository root. Its best matches are screening results, not verified pairings. File presence and dimensions are checked, but the inventory does not validate mask values, lesion-level alignment, checkpoint exposure, or patient identity for every row.

Of 81 rows, 55 confident grading partners occur in the current training manifest and are excluded under this screening rule; 23 are uncertain or outside the current protocol; and three are the existing exploratory candidates. The current manifest is not bound to the checkpoint, so these are manifest-based decisions, not proof of the actual training set. Segmentation-test status does not establish classifier holdout: 13 confident segmentation-test pairs map to grading-training images present in that manifest.

Two further best matches point to grading-test images but remain uncertain: segmentation-test 68 / grading-test 034 (score 0.8521, margin 0.0064, not mutual best), and segmentation-test 70 / grading-test 006 (score 0.9141, margin 0.0301, estimated shift −3,+6 pixels at width 256). Both have four recorded lesion-mask files and matching image dimensions. Neither grading-test ID appears in the existing duplicate-group report. Those facts do not establish a correct match or fine alignment; the 70/006 reduced-resolution difference is substantial, and the 68/034 best match is especially ambiguous. They were not evaluated or tuned here. Other uncertain mappings may hide additional candidates, but the existing evidence does not identify reliable held-out pairs among them.

### Follow-up image and lesion review

The image-only comparison can be reproduced with `.venv/bin/python reports/review_objective3_candidates.py`. Outputs are written under `outputs/explainability/idrid_overlap/candidate_review/`: `IDRiD_68_candidate_comparison.png`, `IDRiD_70_candidate_comparison.png`, `comparison_coordinates.json`, `IDRiD_70_lesion_neighborhoods.png`, and `IDRiD_70_lesion_samples.csv`. The comparison sheets show each segmentation image, its grading-test best match, and the recorded grading-training second-best match. Their optic-disc and vessel crops use the same native image coordinates, centered using the segmentation optic-disc mask; green-channel CLAHE improves visibility. No registration is applied. Different framing can itself make same-coordinate crops look different.

- **68 / 034:** The disc and vessel branching in the magnified views do not establish a match; the second-best 381 also differs. The segmentation image is less sharp, and different framing limits visual comparison. Whether 68 and 034 depict the same eye is uncertain; they do not appear to be the same aligned photograph. Retain exclusion and seek independent correspondence evidence.
- **70 / 006:** Vessel branching and lesion arrangement appear consistent with the same eye, while the image framing and local locations differ. They may be separate captures of that eye, but the images alone do not establish whether this is a different exposure or a transformed version of the same photograph. The second-best 136 does not provide the same obvious lesion correspondence. Patient identity remains unknown.

For 70/006, six large, spatially separated annotated components were selected by descending mask-component area, with centers at least 400 native pixels apart and enough border for the search. The unregistered 480×480 neighborhoods are shown side by side. For a diagnostic only, each 160×160 green-channel segmentation patch was searched within ±200 pixels in the grading image by normalized cross-correlation, without rotation or scaling. The CSV records type, component ID, center, size, search bounds and best offsets. Coordinates are segmentation-image pixels; offsets point from the segmentation center to the best grading-image match:

| Sample | Mask | Center (x, y) | Best offset (x, y) | Correlation |
|---|---|---|---|---:|
| 1 | HE | (1935, 1097) | (−63, 111) | 0.800 |
| 2 | EX | (2314, 1873) | (−87, 120) | 0.485 |
| 3 | SE | (955, 1088) | (10, 0) | 0.557 |
| 4 | HE | (1734, 1535) | (−75, 100) | 0.896 |
| 5 | HE | (1849, 505) | (−39, 101) | 0.760 |
| 6 | EX | (2356, 882) | (−56, 119) | 0.542 |

The visible displacement and varying matches do not establish lesion-level alignment. Low-texture or changed regions can produce misleading correlation peaks; these six samples are not exhaustive and do not validate a transformation. Keep 70/006 excluded from the existing overlap protocol. Any future inclusion would require independently verified correspondence, a prespecified and independently validated alignment method, and lesion-level review; registration cannot prove a pair is correct. Checkpoint training provenance and patient-level independence remain unverified for both pairs. No new Grad-CAM scores were calculated.

The local grading label CSVs contain image name, grade and edema risk, and the inspected local IDRiD metadata does not provide a verified patient identifier. The grading-image duplicate audit uses fixed image-similarity thresholds, not patient linkage or an exhaustive cross-dataset search. A defensible larger study needs an immutable checkpoint-linked training manifest and preprocessing record; source/patient or eye-level identifiers or independently documented patient separation across IDRiD and APTOS; a broader duplicate and near-duplicate audit; verified image correspondence and lesion-level alignment; documented annotation completeness; and additional independently held-out images with lesion masks across relevant grades. Before inspecting their outcomes, specify eligibility, mask policy, registration and thresholds, metrics, and how any sensitivity analysis will be reported.

## Evidence and limits

### Checkpoint provenance audit (2026-11-27)

**Conclusion: the checkpoint's exact training inputs are not verified.** The following evidence is consistent with the current combined-baseline code and manifest, but does not link that specific manifest to the run:

| Local record | SHA-256 | Filesystem modification time (IST, UTC+05:30) |
|---|---|---|
| `outputs/classification/combined_aptos_idrid_baseline/best_model.pth` | `11917bd5cc3669ddecfd9d93fd03b86903520689b809cf6c0381cfabe7dfb64c` | 2026-11-18 21:32:03.980219657 |
| `outputs/classification/splits/combined_train_aptos_idrid.csv` | `b028805c3db7415295018cdacf79f6c3cad658d6c5785f518ba358cdfd62ddcf` | 2026-11-18 20:54:35.815932105 |

The adjacent training history was modified at 21:35:32.889216103 IST on 2026-11-18. Its best row is epoch 9 with validation macro-F1 `0.669902579531528`, exactly matching checkpoint metadata. The history lists ten epochs. Current training code saves a checkpoint at the best validation score and writes history after training, which is consistent with these records and their order. A later validation-analysis file, modified 2026-11-19 20:58:20.063846438 IST, reports the same epoch and macro-F1 for 541 samples. The combined error-analysis report records 2,519 APTOS plus 405 IDRiD training images; those counts match the current manifest. These are corroborating local records, not a cryptographic chain of custody.

The checkpoint contains model and optimizer state, epoch, score, seed, class names and configuration, but no manifest hash, training image list, code revision, or patient identifiers. The current loader reads the current combined manifest; that does not prove the earlier run did so. The tracked manifest in the initial Git commit `282f047` (2026-11-25, after the recorded run) has the same SHA-256 as the present file; Git history and the inspected scripts do not establish its contents at training time. No run-specific manifest snapshot or training log linking the checkpoint to a manifest was identified in the inspected local records. Filesystem times can change, matching configuration and timestamps are circumstantial, and a present-day hash identifies only the current file. The validation and test reports repeat checkpoint metadata and are not independent proof of training inputs.

For the three exploratory grading-test images 091, 088 and 089, the current manifest contains no IDRiD grading-test paths, including these paths. Their mapped segmentation images are visually similar to their grading partners, and the existing IDRiD grading-image duplicate report contains no test entries for those IDs under its specified thresholds. This is an image-level audit finding, not proof that the checkpoint never saw the same or similar image: the actual training list is unknown, the audit is threshold-limited and does not establish cross-dataset or patient-level separation. The inspected local grading-label files contain image names and grades, not verified patient or eye IDs. Patient- and eye-level independence from training therefore remain unverified. The existing evaluation remains **partial and exploratory**, not clinical validation.

Recover a contemporaneous run-specific image manifest or image list and its binding to this checkpoint (for example a retained run log with hashes), along with reliable patient/eye linkage or documented separation. If these records cannot be recovered, retain the provenance limitation and use independently documented held-out data for any future evaluation.

- The selected checkpoint has epoch 9, validation macro-F1 0.669902579531528, seed 42, and APTOS/IDRiD source metadata. SHA-256: `11917bd5cc3669ddecfd9d93fd03b86903520689b809cf6c0381cfabe7dfb64c`. It does not identify a training manifest. Current combined training code loads the current manifest, whose 2,924 rows include no IDRiD grading testing-set paths. This does not prove which data were used for the checkpoint or exclude patient-level or cross-dataset leakage.
- Existing duplicate audit has no test entries for these three IDs. Its candidate correlation threshold is 0.995 at width 256, followed by MAE <= 5 and SSIM >= 0.90; it cannot exclude all near duplicates or same-patient images.
- All six photographs and recorded lesion files were read successfully; each is 4288×2848. Existing masks contain only 0 and 1 and each available mask is nonempty. The SE file for 09 is absent, which is not proof of no soft exudates. All four mask files exist for the other two; this does not prove exhaustive annotations.
- Mapping scores are 0.9986, 0.9997, and 0.9989, with zero shift at width 256. Full-resolution photos are not pixel-identical (mean absolute differences 1.59, 1.81, and 1.84 intensity units). As an additional check, 30, 29, and 28 large annotated microaneurysm components were selected respectively; 30, 29, and 27 64×64 patches fit within the image and were compared within ±4 pixels. One selected patch for 31 was skipped at the boundary. Every usable patch's best match was at zero offset. The earlier report incorrectly counted the skipped patch. This supports, but does not prove, local alignment throughout each image. Previous same-ID ECC artifacts refer to different pairs and were not used.
- Focused tests cover a known intersection, empty activation and lesion union, interpolation and threshold, aggregation, and patch boundaries. Further independent provenance and patient-level audits, lesion-level alignment review, a larger diverse sample, and prespecified sensitivity analyses remain outstanding.
