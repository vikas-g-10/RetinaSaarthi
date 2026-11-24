"""EfficientNet-B0 five-class DR classifier."""

import torch.nn as nn
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
)


def build_model(pretrained=True):
    """
    Build EfficientNet-B0 for five-stage diabetic retinopathy classification.

    Classes:
        0: No DR
        1: Mild
        2: Moderate
        3: Severe
        4: Proliferative DR
    """

    weights = (
        EfficientNet_B0_Weights.DEFAULT if pretrained else None
    )

    model = efficientnet_b0(weights=weights)

    input_features = model.classifier[1].in_features

    model.classifier[1] = nn.Linear(input_features, 5)

    return model
