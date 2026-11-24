# Combined APTOS + IDRiD Model — Error Analysis

## 1. Experiment Details

- Model: EfficientNet-B0
- Training data: 2,519 APTOS + 405 IDRiD images
- Total training images: 2,924
- Validation images: 541
- APTOS test images: 540
- Best checkpoint: Epoch 9
- Evaluation dataset: Held-out APTOS test set

## 2. Overall Classification Performance

| Metric | Result |
|---|---:|
| Accuracy | 84.44% |
| Macro-F1 | 71.66% |

### Per-class F1 scores

| DR Grade | F1-score |
|---|---:|
| No DR | 0.98 |
| Mild | 0.68 |
| Moderate | 0.77 |
| Severe | 0.46 |
| Proliferative DR | 0.68 |

Severe DR was the weakest-performing class.

## 3. Severe DR Error Analysis

Out of 27 Severe DR test images:

- Correctly classified: 13
- Misclassified: 14
- Predicted as Moderate: 10
- Predicted as PDR: 4

## 4. Image Quality Comparison

Median quality metrics for correctly and incorrectly classified Severe DR images:

| Metric | Correct | Misclassified |
|---|---:|---:|
| Sharpness | 12.77 | 9.81 |
| Brightness | 82.55 | 89.03 |
| Contrast | 22.14 | 13.75 |
| Retinal coverage (%) | 83.35 | 83.34 |

## 5. Statistical Analysis

Mann–Whitney U tests compared quality metrics between correctly and incorrectly classified Severe DR images.

| Metric | p-value |
|---|---:|
| Sharpness | 0.1821 |
| Brightness | 0.1148 |
| Contrast | 0.0015 |
| Retinal coverage | 0.5768 |

Contrast showed a statistically significant difference. It remained significant after Bonferroni correction for four comparisons (adjusted p = 0.0060).

## 6. Contrast Analysis Across All Five Grades

Mann–Whitney U tests compared contrast distributions between correct and incorrect predictions separately for each grade.

| Grade | Correct Median | Error Median | Raw p-value | Bonferroni-adjusted p |
|---|---:|---:|---:|---:|
| No DR | 17.7827 | 13.9601 | 0.0051 | 0.0254 |
| Mild | 12.5408 | 12.9281 | 0.3127 | 1.0000 |
| Moderate | 13.8123 | 17.1021 | 0.0022 | 0.0109 |
| Severe | 22.1418 | 13.7508 | 0.0015 | 0.0074 |
| PDR | 15.9947 | 15.7795 | 0.3578 | 1.0000 |

Significant differences were observed for No DR, Moderate DR and Severe DR after correction.

## 7. Contrast Effect Sizes

Cliff's delta was calculated using correctly classified contrast values minus misclassified contrast values.

| Grade | Cliff's delta |
|---|---:|
| No DR | +0.759 |
| Mild | -0.201 |
| Moderate | -0.332 |
| Severe | +0.725 |
| PDR | +0.173 |

The positive Severe DR effect indicates that correctly classified images generally had higher contrast than misclassified images.

The direction of the association differed across grades. Therefore, contrast alone should not be treated as a universal indicator of classification reliability.

## 8. Research Interpretation

The analysis suggests that image contrast may be associated with classification correctness for certain DR grades.

Severe DR images with lower contrast were more frequently misclassified in this test sample. However, Moderate DR showed the opposite direction.

These observations motivate further investigation into the relationship between image quality and model reliability.

## 9. Limitations

- The analysis uses only 540 APTOS test images.
- Only 27 images belong to the Severe DR class.
- Statistical associations do not establish causation.
- Grade-wise comparisons require independent validation.
- No clinical validation has been performed.
- The findings should not be interpreted as evidence that changing image contrast will improve model performance.

## 10. Next Research Step

Evaluate the combined model on the eligible IDRiD test set as an external dataset evaluation.

This will help investigate generalisation beyond the APTOS dataset.
