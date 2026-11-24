from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter

from ml.quality.metrics import compute_metrics


FIXTURE = Path("datasets/aptos/train_images/000c1434d8d7.png")


def measure(tmp_path, image, name):
    path = tmp_path / f"{name}.png"
    image.save(path)
    return compute_metrics(path)


def test_synthetic_quality_degradations_change_expected_metrics(tmp_path):
    image = Image.open(FIXTURE).convert("RGB")
    original = measure(tmp_path, image, "original")
    blurred = measure(tmp_path, image.filter(ImageFilter.GaussianBlur(12)), "blurred")
    dark = measure(tmp_path, ImageEnhance.Brightness(image).enhance(0.15), "dark")
    low_contrast = measure(tmp_path, ImageEnhance.Contrast(image).enhance(0.15), "low_contrast")

    assert blurred.sharpness < original.sharpness
    assert dark.illumination_mean < original.illumination_mean
    assert low_contrast.contrast_p5_p95 < original.contrast_p5_p95
    assert 0.0 < original.field_coverage_frame_percent <= 100.0
