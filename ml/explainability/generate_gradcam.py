
"""Generate Grad-CAM visualizations for three matched IDRiD images.

These images were included in classifier training. Results are exploratory,
not independent validation of localization performance.
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from ml.classification.model import build_model


ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    ROOT
    / "outputs/classification/combined_aptos_idrid_baseline/best_model.pth"
)

GRADE_IMAGE_DIR = (
    ROOT
    / "datasets/idrid/disease_grading/extracted/B. Disease Grading"
    / "1. Original Images/a. Training Set"
)

SEG_IMAGE_DIR = (
    ROOT
    / "datasets/idrid/segmentation/extracted/A. Segmentation"
    / "1. Original Images/a. Training Set"
)

MASK_ROOT = (
    ROOT
    / "datasets/idrid/segmentation/extracted/A. Segmentation"
    / "2. All Segmentation Groundtruths/a. Training Set"
)

OUTPUT_DIR = ROOT / "outputs/explainability/gradcam"

CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative DR",
]

# Exact pixel-matched pairs verified by the matching audit.
MATCHED_IMAGES = [
    {
        "segmentation_id": "IDRiD_01",
        "grading_id": "IDRiD_102",
        "true_grade": 2,
    },
    {
        "segmentation_id": "IDRiD_02",
        "grading_id": "IDRiD_366",
        "true_grade": 3,
    },
    {
        "segmentation_id": "IDRiD_03",
        "grading_id": "IDRiD_369",
        "true_grade": 3,
    },
]

LESION_CATEGORIES = [
    ("1. Microaneurysms", "_MA.tif"),
    ("2. Haemorrhages", "_HE.tif"),
    ("3. Hard Exudates", "_EX.tif"),
    ("4. Soft Exudates", "_SE.tif"),
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model():
    """Load the selected trained EfficientNet-B0 checkpoint."""

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(CHECKPOINT_PATH)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model = build_model(pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    print("Checkpoint:", CHECKPOINT_PATH)
    print("Best epoch:", checkpoint["epoch"])
    print("Best validation macro-F1:", checkpoint["best_val_macro_f1"])
    print("Device:", DEVICE)

    if DEVICE.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    return model


def load_lesion_union(segmentation_id, image_size):
    """Combine available MA, HE, EX and SE masks.

    Missing mask files are not treated as negative annotations.
    """

    combined = np.zeros(
        (image_size[1], image_size[0]),
        dtype=bool,
    )

    available_categories = []

    for folder, suffix in LESION_CATEGORIES:
        mask_path = MASK_ROOT / folder / f"{segmentation_id}{suffix}"

        if not mask_path.is_file():
            continue

        with Image.open(mask_path) as mask_image:
            mask = np.array(mask_image)

        if mask.shape != combined.shape:
            raise ValueError(
                f"Mask dimensions do not match original image: {mask_path}"
            )

        if not np.isin(np.unique(mask), [0, 1]).all():
            raise ValueError(f"Unexpected mask values: {mask_path}")

        combined |= mask == 1
        available_categories.append(suffix.replace(".tif", ""))

    if not available_categories:
        raise RuntimeError(f"No lesion masks found for {segmentation_id}")

    return combined, available_categories


def generate_gradcam(model, image_tensor, target_layer):
    """Generate a Grad-CAM map for the model's predicted class."""

    activations = {}

    def save_activation(module, inputs, output):
        activations["features"] = output

    handle = target_layer.register_forward_hook(save_activation)

    try:
        model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            logits = model(image_tensor)
            predicted_class = int(logits.argmax(dim=1).item())

            feature_maps = activations["features"]

            gradients = torch.autograd.grad(
                outputs=logits[0, predicted_class],
                inputs=feature_maps,
                retain_graph=False,
                create_graph=False,
            )[0]

            weights = gradients.mean(dim=(2, 3), keepdim=True)

            cam = (weights * feature_maps).sum(dim=1, keepdim=True)
            cam = F.relu(cam)

            cam = F.interpolate(
                cam,
                size=(224, 224),
                mode="bilinear",
                align_corners=False,
            )

            cam = cam[0, 0].detach().cpu().numpy()

    finally:
        handle.remove()

    minimum = float(cam.min())
    maximum = float(cam.max())

    if maximum - minimum > 1e-8:
        cam = (cam - minimum) / (maximum - minimum)
    else:
        cam = np.zeros_like(cam)

    probabilities = torch.softmax(logits.detach(), dim=1)[0]
    confidence = float(probabilities[predicted_class].item())

    return cam, predicted_class, confidence


def create_visualization(
    original_image,
    gradcam,
    lesion_mask,
    segmentation_id,
    grading_id,
    true_grade,
    predicted_grade,
    confidence,
    available_categories,
):
    """Save original, Grad-CAM and lesion-mask comparison."""

    width, height = original_image.size

    # Resize Grad-CAM to original photograph dimensions.
    gradcam_image = Image.fromarray(
        np.uint8(np.clip(gradcam, 0, 1) * 255)
    ).resize((width, height), Image.Resampling.BILINEAR)

    gradcam_full = np.asarray(gradcam_image, dtype=np.float32) / 255.0

    original_array = np.asarray(original_image)

    output_path = OUTPUT_DIR / f"{segmentation_id}_gradcam.png"

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    axes[0].imshow(original_array)
    axes[0].set_title("Original fundus image")

    axes[1].imshow(original_array)
    axes[1].imshow(
        gradcam_full,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1,
    )
    axes[1].set_title(
        f"Grad-CAM: {CLASS_NAMES[predicted_grade]}\n"
        f"Confidence: {confidence:.3f}"
    )

    axes[2].imshow(original_array)
    axes[2].imshow(
        lesion_mask,
        cmap="Reds",
        alpha=0.65,
        vmin=0,
        vmax=1,
    )
    axes[2].set_title(
        "Annotated lesion union\n"
        + ", ".join(available_categories)
    )

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        f"{segmentation_id} / {grading_id} | "
        f"True: {CLASS_NAMES[true_grade]} | "
        f"Predicted: {CLASS_NAMES[predicted_grade]}\n"
        "Training image — exploratory visualization, not independent validation",
        fontsize=12,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print("Saved:", output_path)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model()
    target_layer = model.features[-1]

    print("\n========== GRAD-CAM GENERATION ==========")

    for item in MATCHED_IMAGES:
        segmentation_id = item["segmentation_id"]
        grading_id = item["grading_id"]
        true_grade = item["true_grade"]

        grading_path = GRADE_IMAGE_DIR / f"{grading_id}.jpg"
        segmentation_path = SEG_IMAGE_DIR / f"{segmentation_id}.jpg"

        if not grading_path.is_file():
            raise FileNotFoundError(grading_path)

        if not segmentation_path.is_file():
            raise FileNotFoundError(segmentation_path)

        with Image.open(grading_path) as image:
            grading_image = image.convert("RGB")

        with Image.open(segmentation_path) as image:
            segmentation_image = image.convert("RGB")

        if grading_image.size != segmentation_image.size:
            raise ValueError(
                f"Image dimensions differ for {segmentation_id}"
            )

        # Exact pixel matching was already verified. Check again before use.
        if not np.array_equal(
            np.asarray(grading_image),
            np.asarray(segmentation_image),
        ):
            raise ValueError(
                f"Image pixels do not match for {segmentation_id}"
            )

        lesion_mask, categories = load_lesion_union(
            segmentation_id,
            segmentation_image.size,
        )

        image_tensor = INFERENCE_TRANSFORM(grading_image)
        image_tensor = image_tensor.unsqueeze(0).to(DEVICE)

        gradcam, predicted_grade, confidence = generate_gradcam(
            model,
            image_tensor,
            target_layer,
        )

        print(f"\nSegmentation image: {segmentation_id}")
        print(f"Grading image: {grading_id}")
        print(f"True grade: {true_grade} - {CLASS_NAMES[true_grade]}")
        print(
            f"Predicted grade: {predicted_grade} - "
            f"{CLASS_NAMES[predicted_grade]}"
        )
        print(f"Prediction confidence: {confidence:.4f}")
        print("Available lesion masks:", ", ".join(categories))
        print("Annotated lesion pixels:", int(lesion_mask.sum()))

        create_visualization(
            segmentation_image,
            gradcam,
            lesion_mask,
            segmentation_id,
            grading_id,
            true_grade,
            predicted_grade,
            confidence,
            categories,
        )

    print("\nGrad-CAM generation completed.")
    print("Output directory:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
