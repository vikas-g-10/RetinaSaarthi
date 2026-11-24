
"""Custom loss functions for DR classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassWeightedFocalLoss(nn.Module):
    """Multi-class focal loss with optional class weights."""

    def __init__(self, class_weights=None, gamma=2.0):
        super().__init__()

        self.gamma = gamma

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits, targets):
        log_probabilities = F.log_softmax(logits, dim=1)

        log_pt = log_probabilities.gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)

        pt = log_pt.exp()

        focal_factor = (1.0 - pt).pow(self.gamma)

        loss = -focal_factor * log_pt

        if self.class_weights is not None:
            alpha = self.class_weights[targets]
            loss = alpha * loss

        return loss.mean()
