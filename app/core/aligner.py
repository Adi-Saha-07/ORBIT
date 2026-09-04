"""
Image Alignment Module for ORBIT (Stage 2).

Uses ORB (Oriented FAST and Rotated BRIEF) keypoint detection + RANSAC
Homography estimation to co-register two temporal satellite captures into the
same geometric coordinate space.
"""

import cv2
import numpy as np
from app.core.preprocessor import to_grayscale, match_dimensions

def align_images_orb(
    img_reference: np.ndarray,
    img_target: np.ndarray,
    max_features: int = 4000,
    keep_percent: float = 0.35,
    reproj_threshold: float = 4.0,
) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """
    Aligns the target image (T-1) to the reference frame (T-0) using ORB and Homography.

    Args:
        img_reference: Base image array (T-0 baseline) in BGR.
        img_target: Target image array (T-1 temporal capture) in BGR.
        max_features: Number of ORB keypoints to detect.
        keep_percent: Top percentage of confident feature matches to retain.
        reproj_threshold: Maximum allowed reprojection error in pixels for RANSAC.

    Returns:
        tuple (aligned_target: np.ndarray, homography_matrix: np.ndarray | None, telemetry: dict)
    """
    # 1. Ensure compatible canvas dimensions first
    img_reference, img_target = match_dimensions(img_reference, img_target)
    h_ref, w_ref = img_reference.shape[:2]

    # Convert to grayscale for feature extraction
    gray_ref = to_grayscale(img_reference)
    gray_tgt = to_grayscale(img_target)

    # 2. Initialize ORB detector
    orb = cv2.ORB_create(
        nfeatures=max_features,
        scaleFactor=1.2,
        nlevels=8,
        edgeThreshold=15,
        firstLevel=0,
        WTA_K=2,
        scoreType=cv2.ORB_HARRIS_SCORE,
        patchSize=31,
        fastThreshold=20,
    )

    # Detect keypoints and compute descriptors
    kp_ref, des_ref = orb.detectAndCompute(gray_ref, None)
    kp_tgt, des_tgt = orb.detectAndCompute(gray_tgt, None)

    telemetry = {
        "features_reference": len(kp_ref) if kp_ref is not None else 0,
        "features_target": len(kp_tgt) if kp_tgt is not None else 0,
        "matches_total": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "aligned": False,
        "homography_matrix": None,
        "method": "ORB_RANSAC_HOMOGRAPHY",
    }

    # If insufficient features were found in either image
    if des_ref is None or des_tgt is None or len(des_ref) < 4 or len(des_tgt) < 4:
        telemetry["status"] = "FALLBACK_INSUFFICIENT_FEATURES"
        return img_target, None, telemetry

    # 3. Match features using Hamming distance with cross-check
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des_tgt, des_ref)

    # Sort matches by distance (lower distance = higher confidence)
    matches = sorted(matches, key=lambda x: x.distance)

    # Retain the top confident matches
    num_good_matches = max(int(len(matches) * keep_percent), 4)
    good_matches = matches[:num_good_matches]
    telemetry["matches_total"] = len(good_matches)

    if len(good_matches) < 4:
        telemetry["status"] = "FALLBACK_TOO_FEW_MATCHES"
        return img_target, None, telemetry

    # 4. Extract matched coordinates
    pts_tgt = np.float32([kp_tgt[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    pts_ref = np.float32([kp_ref[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 5. Estimate Homography Matrix with RANSAC
    H, mask = cv2.findHomography(pts_tgt, pts_ref, cv2.RANSAC, reproj_threshold)

    if H is None or mask is None:
        telemetry["status"] = "FALLBACK_HOMOGRAPHY_FAILED"
        return img_target, None, telemetry

    inliers_count = int(np.sum(mask))
    inlier_ratio = round(inliers_count / len(good_matches), 3) if good_matches else 0.0

    telemetry["inliers"] = inliers_count
    telemetry["inlier_ratio"] = inlier_ratio
    telemetry["homography_matrix"] = H.round(4).tolist()

    # If inliers are too low, transformation is unstable; fallback to unwarped target
    if inliers_count < 4 or inlier_ratio < 0.15:
        telemetry["status"] = "FALLBACK_LOW_INLIER_CONSENSUS"
        return img_target, H, telemetry

    # 6. Apply perspective transformation warp
    aligned_target = cv2.warpPerspective(
        img_target,
        H,
        (w_ref, h_ref),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    telemetry["aligned"] = True
    telemetry["status"] = "ALIGNMENT_LOCKED"

    return aligned_target, H, telemetry
