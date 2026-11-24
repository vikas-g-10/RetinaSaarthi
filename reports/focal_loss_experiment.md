# Class-Weighted Focal Loss Experiment

## 1. Objective

To investigate whether class-weighted focal loss improves five-class diabetic retinopathy grading, particularly for underrepresented grades such as Mild, Severe, and Proliferative DR.

## 2. Experimental Configuration

- Architecture: EfficientNet-B0
- Training datasets: APTOS and IDRiD
- Training images: 2,924
- Validation images: 541
- Test images: 540 (not used for this experiment's model selection)
- Input size: 224 × 224
- Batch size: 16
- Optimizer: AdamW
- Learning rate: 0.0001
- Maximum epochs: 10
- Early stopping patience: 3
- Random seed: 42
- Focal loss gamma: 2.0
- Class weighting: Inverse-frequency weights calculated from the combined training dataset

The experiment retained the original combined model's architecture, dataset splits, and training configuration. The loss function was changed to class-weighted focal loss.

## 3. Results

Best checkpoint: Epoch 10

| Metric | Result |
|---|---:|
| Validation accuracy | 80.22% |
| Validation macro-F1 | 64.13% |
| Mean confidence | 0.7895 |
| Mean absolute grade error | 0.2606 |
| Within-one-grade accuracy | 94.27% |
| Errors of two or more grades | 31 |
| High-confidence incorrect predictions (≥80%) | 12 |

### Per-class performance

| Grade | Precision | Recall | F1-score |
|---|---:|---:|---:|
| No DR | 0.97 | 0.98 | 0.98 |
| Mild | 0.62 | 0.63 | 0.62 |
| Moderate | 0.73 | 0.69 | 0.71 |
| Severe | 0.30 | 0.37 | 0.33 |
| Proliferative DR | 0.56 | 0.57 | 0.56 |

## 4. Comparison with Original Combined Model

| Metric | Original combined model | Focal-loss model |
|---|---:|---:|
| Validation accuracy | 82.81% | 80.22% |
| Validation macro-F1 | 66.99% | 64.13% |
| Mean absolute grade error | 0.2274 | 0.2606 |
| Within-one-grade accuracy | 95.19% | 94.27% |
| Errors of two or more grades | 26 | 31 |

### Per-class F1 comparison

| Grade | Original combined | Focal loss |
|---|---:|---:|
| No DR | 0.97 | 0.98 |
| Mild | 0.64 | 0.62 |
| Moderate | 0.77 | 0.71 |
| Severe | 0.38 | 0.33 |
| Proliferative DR | 0.58 | 0.56 |

## 5. Interpretation

The focal-loss model achieved a validation macro-F1 of 64.13%, compared with 66.99% for the original combined model.

Although No DR classification improved slightly, F1 decreased for Mild, Moderate, Severe, and Proliferative DR. The focal-loss experiment also produced more errors involving grade differences of two or more.

Training macro-F1 reached 85.23%, while validation macro-F1 reached 64.13%. This gap suggests overfitting.

## 6. Conclusion

Class-weighted focal loss with gamma = 2.0 did not outperform the original weighted cross-entropy configuration on the validation dataset.

The original combined EfficientNet-B0 model remains the selected classifier. The focal-loss model is retained as a documented experimental comparison rather than adopted as the primary model.

The held-out test partition was not used to tune or select this experiment.
