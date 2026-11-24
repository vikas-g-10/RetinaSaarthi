
"""Aspect-ratio-preserving transforms for the letterbox experiment."""

from PIL import Image, ImageOps
from torchvision import transforms


class Letterbox:
    """Resize an image proportionally and center it on a square canvas."""

    def __init__(self, size=224, fill=(0, 0, 0)):
        self.size = size
        self.fill = fill

    def __call__(self, image):
        image = ImageOps.contain(
            image,
            (self.size, self.size),
            method=Image.Resampling.BILINEAR,
        )

        canvas = Image.new(
            "RGB",
            (self.size, self.size),
            self.fill,
        )

        x = (self.size - image.width) // 2
        y = (self.size - image.height) // 2

        canvas.paste(image, (x, y))

        return canvas


def get_letterbox_transforms(training=False):
    """Return letterbox preprocessing for training or evaluation."""

    if training:
        return transforms.Compose([
            Letterbox(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    return transforms.Compose([
        Letterbox(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
