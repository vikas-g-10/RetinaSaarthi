
"""Standardised fundus-image quality metrics.

The field mask is acquisition-oriented rather than pathology-oriented.
It identifies the illuminated retinal field using grayscale thresholding,
morphological operations, and the largest connected component.

Metrics are calculated after cropping to the detected field and resizing
to STANDARD_SIZE. This is an engineering standardisation choice, not a
clinically validated specification.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

STANDARD_SIZE = 512
WORKING_MAX_DIMENSION = 1024


@dataclass(frozen=True)
class QualityMetrics:
    width: int
    height: int
    pixel_count: int
    sharpness: float
    illumination_mean: float
    illumination_std: float
    illumination_quadrant_ratio: float
    contrast_p5_p95: float
    field_coverage_percent: float
    field_coverage_frame_percent: float
    field_touches_edge: bool
    touches_top: bool
    touches_bottom: bool
    touches_left: bool
    touches_right: bool


def _resize_max_dimension(
    image: np.ndarray,
    max_dimension: int,
) -> np.ndarray:
    """Resize an image while preserving its aspect ratio."""
    height, width = image.shape[:2]
    longest_side = max(height, width)

    if longest_side <= max_dimension:
        return image.copy()

    scale = max_dimension / longest_side
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    return cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def find_field_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Detect the illuminated fundus field and return a binary mask."""
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Input image is empty.")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    work = _resize_max_dimension(gray, WORKING_MAX_DIMENSION)

    threshold = max(
        5,
        int(np.percentile(work, 1)),
    )

    candidate = (work > threshold).astype(np.uint8)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (9, 9),
    )

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        kernel,
    )

    candidate = cv2.morphologyEx(
        candidate,
        cv2.MORPH_OPEN,
        kernel,
    )

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate,
        connectivity=8,
    )

    if count <= 1:
        raise ValueError("Could not detect a retinal field.")

    largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    mask = (labels == largest_component).astype(np.uint8)

    if int(mask.sum()) < 100:
        raise ValueError("Detected retinal field is too small.")

    original_height, original_width = image_bgr.shape[:2]

    if mask.shape != (original_height, original_width):
        mask = cv2.resize(
            mask,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        )

    return mask.astype(bool)


def _crop_resize(
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop to the detected field bounding box and resize to standard size."""
    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("Field mask is empty.")

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1

    cropped_image = image[y0:y1, x0:x1]
    cropped_mask = mask[y0:y1, x0:x1].astype(np.uint8)

    resized_image = cv2.resize(
        cropped_image,
        (STANDARD_SIZE, STANDARD_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    resized_mask = cv2.resize(
        cropped_mask,
        (STANDARD_SIZE, STANDARD_SIZE),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    return resized_image, resized_mask


def compute_metrics(image_path: str | Path) -> QualityMetrics:
    """Calculate standardised quality and field-framing metrics."""
    image_path = Path(image_path)

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            f"OpenCV could not decode image: {image_path}"
        )

    height, width = image.shape[:2]
    pixel_count = width * height

    working = _resize_max_dimension(
        image,
        WORKING_MAX_DIMENSION,
    )

    working_height, working_width = working.shape[:2]

    mask = find_field_mask(working)

    ys, xs = np.where(mask)

    if len(xs) == 0 or len(ys) == 0:
        raise ValueError(
            f"Could not detect retinal field: {image_path}"
        )

    # Directional edge-contact indicators.
    touches_top = bool(ys.min() == 0)
    touches_bottom = bool(ys.max() == working_height - 1)
    touches_left = bool(xs.min() == 0)
    touches_right = bool(xs.max() == working_width - 1)

    touches_edge = bool(
        touches_top
        or touches_bottom
        or touches_left
        or touches_right
    )

    # This is mask area divided by its bounding-box area.
    # It is NOT the percentage of the original frame occupied by the retina.
    bbox_area = (
        (int(xs.max()) - int(xs.min()) + 1)
        * (int(ys.max()) - int(ys.min()) + 1)
    )

    field_coverage_percent = (
        100.0 * float(mask.sum()) / float(bbox_area)
    )
    field_coverage_frame_percent = (
        100.0 * float(mask.sum()) / float(mask.size)
    )

    resized, resized_mask = _crop_resize(
        working,
        mask,
    )

    gray = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2GRAY,
    )

    green = resized[:, :, 1]

    valid_gray = gray[resized_mask]
    valid_green = green[resized_mask]

    if valid_gray.size == 0 or valid_green.size == 0:
        raise ValueError(
            f"No valid pixels inside retinal field: {image_path}"
        )

    # Sharpness: variance of the Laplacian on the green channel.
    laplacian = cv2.Laplacian(
        green,
        cv2.CV_64F,
    )

    sharpness = float(
        np.var(laplacian[resized_mask])
    )

    # Illumination.
    illumination_mean = float(
        np.mean(valid_gray)
    )

    illumination_std = float(
        np.std(valid_gray)
    )

    # Illumination variation across four quadrants.
    half = STANDARD_SIZE // 2

    quadrants = [
        (slice(0, half), slice(0, half)),
        (slice(0, half), slice(half, STANDARD_SIZE)),
        (slice(half, STANDARD_SIZE), slice(0, half)),
        (slice(half, STANDARD_SIZE), slice(half, STANDARD_SIZE)),
    ]

    quadrant_means = []

    for y_slice, x_slice in quadrants:
        quadrant_mask = resized_mask[y_slice, x_slice]
        quadrant_gray = gray[y_slice, x_slice]

        values = quadrant_gray[quadrant_mask]

        if values.size > 0:
            quadrant_means.append(
                float(np.mean(values))
            )

    if len(quadrant_means) >= 2:
        darkest = min(quadrant_means)
        brightest = max(quadrant_means)

        illumination_quadrant_ratio = (
            brightest / max(darkest, 1e-6)
        )
    else:
        illumination_quadrant_ratio = float("nan")

    # Contrast: 95th percentile minus 5th percentile.
    p5, p95 = np.percentile(
        valid_gray,
        [5, 95],
    )

    contrast_p5_p95 = float(p95 - p5)

    return QualityMetrics(
        width=width,
        height=height,
        pixel_count=pixel_count,
        sharpness=sharpness,
        illumination_mean=illumination_mean,
        illumination_std=illumination_std,
        illumination_quadrant_ratio=illumination_quadrant_ratio,
        contrast_p5_p95=contrast_p5_p95,
        field_coverage_percent=field_coverage_percent,
        field_coverage_frame_percent=field_coverage_frame_percent,
        field_touches_edge=touches_edge,
        touches_top=touches_top,
        touches_bottom=touches_bottom,
        touches_left=touches_left,
        touches_right=touches_right,
    )


def make_mask_overlay(
    image_path: str | Path,
) -> np.ndarray:
    """Return a high-visibility fundus field-mask review overlay."""
    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise ValueError(
            f"OpenCV could not decode image: {image_path}"
        )

    mask = find_field_mask(image)

    # Semi-transparent cyan fill highlights the detected field.
    overlay = image.copy()

    cyan = np.array(
        [255, 255, 0],
        dtype=np.float32,
    )

    overlay[mask] = (
        0.65 * image[mask].astype(np.float32)
        + 0.35 * cyan
    ).astype(np.uint8)

    # Bright green contour makes the detected boundary easier to inspect.
    contours, _ = cv2.findContours(
        mask.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (0, 255, 0),
        thickness=3,
    )

    return overlay
