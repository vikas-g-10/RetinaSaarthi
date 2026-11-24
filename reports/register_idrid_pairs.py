
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

# --------------------------------------------------
# 1. Dataset paths
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

output_root = Path("reports/idrid_registration")
output_root.mkdir(parents=True, exist_ok=True)

image_ids = [1, 9]

# --------------------------------------------------
# 2. ORB feature detector
# --------------------------------------------------

orb = cv2.ORB_create(
    nfeatures=10000,
    scaleFactor=1.2,
    nlevels=8,
    edgeThreshold=31,
    fastThreshold=10
)

matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

results = []

# --------------------------------------------------
# 3. Process each image pair
# --------------------------------------------------

for image_id in image_ids:

    print("\n" + "=" * 60)
    print(f"REGISTERING IDRiD {image_id:02d}")
    print("=" * 60)

    seg_path = seg_root / f"IDRiD_{image_id:02d}.jpg"
    grading_path = grading_root / f"IDRiD_{image_id:03d}.jpg"

    reference = cv2.imread(str(seg_path))
    moving = cv2.imread(str(grading_path))

    if reference is None or moving is None:
        print("ERROR: Could not read one or both images.")
        continue

    if reference.shape != moving.shape:
        print("ERROR: Image dimensions do not match.")
        continue

    # Grayscale images for feature detection
    ref_gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    mov_gray = cv2.cvtColor(moving, cv2.COLOR_BGR2GRAY)

    # CLAHE helps feature detection under illumination differences.
    # Original images remain unchanged.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    ref_enhanced = clahe.apply(ref_gray)
    mov_enhanced = clahe.apply(mov_gray)

    kp_ref, des_ref = orb.detectAndCompute(ref_enhanced, None)
    kp_mov, des_mov = orb.detectAndCompute(mov_enhanced, None)

    if des_ref is None or des_mov is None:
        print("ERROR: Not enough ORB features.")
        continue

    print("Reference keypoints:", len(kp_ref))
    print("Moving keypoints:", len(kp_mov))

    # Find two nearest descriptor matches
    raw_matches = matcher.knnMatch(
        des_mov, des_ref, k=2
    )

    # Lowe ratio test
    good_matches = []

    for pair in raw_matches:
        if len(pair) == 2:
            first, second = pair

            if first.distance < 0.75 * second.distance:
                good_matches.append(first)

    print("Good matches:", len(good_matches))

    if len(good_matches) < 10:
        print("ERROR: Not enough reliable matches.")
        continue

    # Source = disease-grading image
    # Destination = segmentation reference image
    src_points = np.float32([
        kp_mov[m.queryIdx].pt for m in good_matches
    ]).reshape(-1, 1, 2)

    dst_points = np.float32([
        kp_ref[m.trainIdx].pt for m in good_matches
    ]).reshape(-1, 1, 2)

    # Estimate rotation + translation + uniform scale
    matrix, inlier_mask = cv2.estimateAffinePartial2D(
        src_points,
        dst_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=10000,
        confidence=0.99,
        refineIters=10
    )

    if matrix is None or inlier_mask is None:
        print("ERROR: RANSAC could not estimate a transformation.")
        continue

    inliers = inlier_mask.ravel().astype(bool)
    inlier_count = int(np.sum(inliers))
    inlier_ratio = inlier_count / len(good_matches)

    print("RANSAC inliers:", inlier_count)
    print("Inlier ratio:", round(inlier_ratio, 4))
    print("\nTransformation matrix:")
    print(matrix)

    # Calculate transformation properties
    a, b = matrix[0, 0], matrix[0, 1]
    c, d = matrix[1, 0], matrix[1, 1]

    scale = np.sqrt(a * a + c * c)
    rotation = np.degrees(np.arctan2(c, a))

    tx = matrix[0, 2]
    ty = matrix[1, 2]

    print("Estimated scale:", round(scale, 6))
    print("Estimated rotation (degrees):", round(rotation, 4))
    print("Estimated translation X:", round(tx, 2))
    print("Estimated translation Y:", round(ty, 2))

    # Warp grading image into segmentation coordinates
    height, width = reference.shape[:2]

    registered = cv2.warpAffine(
        moving,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    # Create comparison images
    overlay = cv2.addWeighted(
        reference, 0.5, registered, 0.5, 0
    )

    difference = cv2.absdiff(reference, registered)

    # Draw feature matches, showing RANSAC inliers only
    inlier_matches = [
        match for match, keep in zip(good_matches, inliers)
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

    # Save outputs
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

    cv2.imwrite(
        str(output_root / f"idrid_{image_id:02d}_matches.jpg"),
        match_visual
    )

    results.append({
        "image_id": image_id,
        "reference_keypoints": len(kp_ref),
        "moving_keypoints": len(kp_mov),
        "good_matches": len(good_matches),
        "ransac_inliers": inlier_count,
        "inlier_ratio": inlier_ratio,
        "scale": scale,
        "rotation_degrees": rotation,
        "translation_x": tx,
        "translation_y": ty
    })

    print("Saved registered image, overlay, difference and matches.")

# --------------------------------------------------
# 4. Save experiment summary
# --------------------------------------------------

if results:

    df = pd.DataFrame(results)

    summary_path = output_root / "registration_summary.csv"
    df.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print("REGISTRATION EXPERIMENT SUMMARY")
    print("=" * 60)

    print(df.to_string(index=False))
    print("\nSummary saved to:", summary_path)

else:
    print("\nNo image pairs were successfully registered.")
