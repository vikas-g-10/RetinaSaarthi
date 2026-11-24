
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

base = Path("datasets/idrid")

seg_root = (
    base / "segmentation/extracted/"
    "A. Segmentation/1. Original Images/a. Training Set"
)

grading_root = (
    base / "disease_grading/extracted/"
    "B. Disease Grading/1. Original Images/a. Training Set"
)

output_root = Path("reports/idrid_registration_sift")
output_root.mkdir(parents=True, exist_ok=True)

image_ids = [1, 9]

# --------------------------------------------------
# 2. SIFT detector and matcher
# --------------------------------------------------

sift = cv2.SIFT_create(
    nfeatures=10000,
    contrastThreshold=0.02,
    edgeThreshold=15
)

bf = cv2.BFMatcher(cv2.NORM_L2)

results = []


def detect_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Improve local contrast for feature detection
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    # Ignore black background
    mask = np.where(gray > 10, 255, 0).astype(np.uint8)

    keypoints, descriptors = sift.detectAndCompute(
        enhanced, mask
    )

    return keypoints, descriptors


def ratio_matches(des1, des2):
    pairs = bf.knnMatch(des1, des2, k=2)

    good = []

    for pair in pairs:
        if len(pair) == 2:
            first, second = pair

            if first.distance < 0.75 * second.distance:
                good.append(first)

    return good


# --------------------------------------------------
# 3. Process image pairs
# --------------------------------------------------

for image_id in image_ids:

    print("\n" + "=" * 65)
    print(f"SIFT REGISTRATION: IDRiD {image_id:02d}")
    print("=" * 65)

    seg_path = seg_root / f"IDRiD_{image_id:02d}.jpg"
    grading_path = grading_root / f"IDRiD_{image_id:03d}.jpg"

    reference = cv2.imread(str(seg_path))
    moving = cv2.imread(str(grading_path))

    if reference is None or moving is None:
        print("ERROR: Could not read images.")
        continue

    if reference.shape != moving.shape:
        print("ERROR: Image dimensions differ.")
        continue

    height, width = reference.shape[:2]

    kp_ref, des_ref = detect_features(reference)
    kp_mov, des_mov = detect_features(moving)

    if des_ref is None or des_mov is None:
        print("ERROR: No descriptors detected.")
        continue

    print("Reference keypoints:", len(kp_ref))
    print("Moving keypoints:", len(kp_mov))

    # Match moving -> reference
    matches_forward = ratio_matches(des_mov, des_ref)

    # Match reference -> moving
    matches_reverse = ratio_matches(des_ref, des_mov)

    # Keep only mutual matches
    reverse_pairs = {
        (m.trainIdx, m.queryIdx)
        for m in matches_reverse
    }

    mutual_matches = [
        m for m in matches_forward
        if (m.queryIdx, m.trainIdx) in reverse_pairs
    ]

    print("Forward ratio matches:", len(matches_forward))
    print("Reverse ratio matches:", len(matches_reverse))
    print("Mutual matches:", len(mutual_matches))

    if len(mutual_matches) < 6:
        print("REJECTED: Too few mutual matches.")
        continue

    # Source = grading image
    # Destination = segmentation image
    src_points = np.float32([
        kp_mov[m.queryIdx].pt for m in mutual_matches
    ]).reshape(-1, 1, 2)

    dst_points = np.float32([
        kp_ref[m.trainIdx].pt for m in mutual_matches
    ]).reshape(-1, 1, 2)

    # Estimate partial affine transformation
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        src_points,
        dst_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=20000,
        confidence=0.99,
        refineIters=20
    )

    if matrix is None or inlier_mask is None:
        print("REJECTED: RANSAC could not estimate transformation.")
        continue

    inliers = inlier_mask.ravel().astype(bool)
    inlier_count = int(np.sum(inliers))
    inlier_ratio = inlier_count / len(mutual_matches)

    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]

    scale = np.sqrt(a * a + c * c)
    rotation = np.degrees(np.arctan2(c, a))

    tx = matrix[0, 2]
    ty = matrix[1, 2]

    # Reprojection error for RANSAC inliers
    projected = cv2.transform(src_points, matrix)

    errors = np.linalg.norm(
        projected.reshape(-1, 2) - dst_points.reshape(-1, 2),
        axis=1
    )

    mean_error = float(np.mean(errors[inliers]))
    median_error = float(np.median(errors[inliers]))

    print("\nRANSAC inliers:", inlier_count)
    print("Inlier ratio:", round(inlier_ratio, 4))
    print("Mean inlier reprojection error:", round(mean_error, 3))
    print("Median inlier reprojection error:", round(median_error, 3))

    print("\nTransformation matrix:")
    print(matrix)

    print("Scale:", round(scale, 6))
    print("Rotation:", round(rotation, 4), "degrees")
    print("Translation X:", round(tx, 2))
    print("Translation Y:", round(ty, 2))

    # Conservative experimental acceptance checks.
    # These are screening guardrails, not universal thresholds.
    plausible = (
        inlier_count >= 10
        and inlier_ratio >= 0.20
        and 0.80 <= scale <= 1.20
        and abs(rotation) <= 5.0
        and mean_error <= 3.0
    )

    print("\nTransformation screening:",
          "PLAUSIBLE CANDIDATE" if plausible else "REJECTED / NEEDS REVIEW")

    # Draw only RANSAC inlier matches
    inlier_matches = [
        match for match, keep in zip(mutual_matches, inliers)
        if keep
    ]

    match_visual = cv2.drawMatches(
        moving,
        kp_mov,
        reference,
        kp_ref,
        inlier_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    cv2.imwrite(
        str(output_root / f"idrid_{image_id:02d}_matches.jpg"),
        match_visual
    )

    # Warp only when screening checks pass
    if plausible:

        registered = cv2.warpAffine(
            moving,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )

        overlay = cv2.addWeighted(
            reference, 0.5, registered, 0.5, 0
        )

        difference = cv2.absdiff(reference, registered)

        cv2.imwrite(
            str(output_root / f"idrid_{image_id:02d}_registered.jpg"),
            registered
        )

        cv2.imwrite(
            str(output_root / f"idrid_{image_id:02d}_overlay.jpg"),
            overlay
        )

        cv2.imwrite(
            str(output_root / f"idrid_{image_id:02d}_difference.jpg"),
            difference
        )

        print("Candidate registered image and overlay saved.")
    else:
        print("No registered image generated because the transformation failed screening.")

    results.append({
        "image_id": image_id,
        "reference_keypoints": len(kp_ref),
        "moving_keypoints": len(kp_mov),
        "forward_matches": len(matches_forward),
        "reverse_matches": len(matches_reverse),
        "mutual_matches": len(mutual_matches),
        "ransac_inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        "mean_reprojection_error": mean_error,
        "median_reprojection_error": median_error,
        "scale": scale,
        "rotation_degrees": rotation,
        "translation_x": tx,
        "translation_y": ty,
        "plausible_candidate": plausible
    })


# --------------------------------------------------
# 4. Save results
# --------------------------------------------------

if results:

    df = pd.DataFrame(results)

    summary_path = output_root / "sift_registration_summary.csv"
    df.to_csv(summary_path, index=False)

    print("\n" + "=" * 65)
    print("SIFT REGISTRATION SUMMARY")
    print("=" * 65)

    print(df.to_string(index=False))
    print("\nSaved:", summary_path)

else:
    print("\nNo transformations could be estimated.")
