
"""EfficientNet-B0 with a rank-consistent ordinal DR classification head."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
)


class OrdinalHead(nn.Module):
    """Predict four ordered cumulative DR-grade decisions."""

    def __init__(self, in_features):
        super().__init__()

        self.score = nn.Linear(in_features, 1, bias=False)

        # Increasing thresholds guarantee non-increasing ordinal logits.
        self.base_threshold = nn.Parameter(torch.tensor(0.10))
        self.threshold_deltas = nn.Parameter(
            torch.tensor([0.18, 1.25, 0.65])
        )

    def forward(self, features):
        score = self.score(features)

        thresholds = torch.cat([
            self.base_threshold.unsqueeze(0),
            self.base_threshold.unsqueeze(0)
            + torch.cumsum(F.softplus(self.threshold_deltas), dim=0),
        ])

        return score - thresholds.unsqueeze(0)


class OrdinalEfficientNet(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = (
            EfficientNet_B0_Weights.DEFAULT if pretrained else None
        )

        self.backbone = efficientnet_b0(weights=weights)

        input_features = self.backbone.classifier[1].in_features

        self.backbone.classifier[1] = nn.Identity()
        self.ordinal_head = OrdinalHead(input_features)

    def forward(self, images):
        features = self.backbone(images)
        return self.ordinal_head(features)


def build_ordinal_model(pretrained=True):
    return OrdinalEfficientNet(pretrained=pretrained)


def ordinal_targets(labels):
    """Convert grades 0–4 into four cumulative binary targets."""
    thresholds = torch.arange(4, device=labels.device)
    return (labels.unsqueeze(1) > thresholds).float()


def decode_ordinal_logits(logits):
    """Convert four cumulative decisions into grades 0–4."""
    probabilities = torch.sigmoid(logits)
    return (probabilities >= 0.5).sum(dim=1)
