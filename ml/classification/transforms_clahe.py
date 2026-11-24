
"""CLAHE preprocessing for the RetinaSaarthi experiment."""

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms


class ApplyCLAHE:
    """Apply CLAHE to luminance while preserving LAB colour channels."""

    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, image):
        # Convert PIL RGB image to NumPy RGB array.
        rgb = np.array(image.convert("RGB"))

        # Convert RGB to LAB colour space.
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

        # Separate luminance and colour information.
        luminance, channel_a, channel_b = cv2.split(lab)

        # Apply CLAHE only to luminance.
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.tile_grid_size,
        )

        enhanced_luminance = clahe.apply(luminance)

        # Recombine LAB channels and convert back to RGB.
        enhanced_lab = cv2.merge(
            (enhanced_luminance, channel_a, channel_b)
        )

        enhanced_rgb = cv2.cvtColor(
            enhanced_lab,
            cv2.COLOR_LAB2RGB,
        )

        return Image.fromarray(enhanced_rgb)


def get_clahe_transforms(training=False, clip_limit=1.0):
    clahe = ApplyCLAHE(
        clip_limit=clip_limit,
        tile_grid_size=(8, 8)
    )

    if training:
        return transforms.Compose([
            clahe,
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    return transforms.Compose([
        clahe,
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
