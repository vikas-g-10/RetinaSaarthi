# Objective 3: final exploratory prototype study

**Status:** The prototype implementation and this three-image exploratory study are complete. Independent validation is pending. RetinaSaarthi is an academic research prototype, not a clinically validated diagnostic device.

## Method

We used the existing EfficientNet-B0 checkpoint and three mapped IDRiD segmentation/grading image pairs. The grading images all have reference grade 2. The existing evaluation resized each grading image to 224 × 224 and applied ImageNet normalization, then generated Grad-CAM for the model's predicted class at `features[-1]`. Each map was normalized per image, bilinearly resized to the original image dimensions, and thresholded at the fixed value 0.5. No registration was applied; sampled local alignment checks support direct comparison but do not prove perfect alignment.

The reference was the union of available microaneurysm, haemorrhage, hard-exudate and soft-exudate masks; the optic disc was excluded. A missing mask was treated as unknown. Intersection-over-union (IoU) and Dice measure overlap between the thresholded map and recorded union. Lesion coverage is the fraction of recorded lesion pixels within the thresholded map. See the [protocol](objective3_protocol.md) for exact definitions and the [detailed results](objective3_results.md) for evidence and reproduction instructions.

## Results

| Segmentation / grading image | Predicted grade | Available masks | IoU | Dice | Lesion coverage |
|---|---:|---|---:|---:|---:|
| 09 / 091 | 2 | MA, HE, EX; SE missing | 0.0902 | 0.1655 | 0.4030 |
| 22 / 088 | 3 | MA, HE, EX, SE | 0.0533 | 0.1012 | 0.3418 |
| 31 / 089 | 4 | MA, HE, EX, SE | 0.0360 | 0.0696 | 0.2063 |
| Mean of 22 and 31, with four mask files present | — | — | 0.0447 | 0.0854 | 0.2741 |

The mean is unweighted and excludes 09 because its mask set is incomplete. Four files being present does not prove that every lesion is annotated.

## Visual interpretation

The review figures use yellow for overlap, orange for recorded lesion pixels outside the thresholded map, and blue for highlighted pixels outside the available masks. Blue does not establish that a region is lesion-free; annotations may be incomplete. The figures are downsampled, so small lesions may be difficult to see.

- **09 / 091:** The highlighted region crosses part of the central lesion cluster, while many recorded lesions remain outside it. Highlighting also extends beyond the available masks.
- **22 / 088:** Some recorded lesions overlap the highlighted region, but other recorded lesions, including areas toward the right and lower image, lie outside it. A substantial highlighted area lies outside the available masks.
- **31 / 089:** The highlighted region intersects part of the central recorded lesions; much of the recorded union and much of the highlighted area do not overlap.

These are observations of spatial overlap. They do not explain the causes of the model's predictions or errors.

## Limitations and next steps

Only three images were examined, all with reference grade 2; the findings cannot be generalized to other images or grades. The SE mask for 09 is missing, which does not mean soft exudates are absent. The checkpoint's exact training manifest and patient- or eye-level independence are unverified, so these cases are not proven held out. The single 0.5 threshold is arbitrary and was not optimized for clinical use. Images 22 and 31 show predicted-class maps for grades 3 and 4, respectively, while their reference grade is 2. Alignment and annotation completeness also remain uncertain.

Grad-CAM overlap does not establish clinical correctness or clinical validation. Independent validation remains pending and would require documented training provenance and patient-level separation, verified image/mask alignment, and a larger diverse independently held-out set with a prespecified evaluation protocol.
