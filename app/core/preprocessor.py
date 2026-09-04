"""
Image Preprocessing Module for ORBIT.

Provides image loading, dimension normalization, and color space transformations
before feature alignment and structural change detection.
"""

import os
import cv2
import numpy as np

def load_image_bgr(image_path: str) -> np.ndarray:
    """
    Safely loads an image from disk in BGR format using OpenCV.
    Raises FileNotFoundError or ValueError if image cannot be decoded.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Using imdecode with numpy buffer to avoid Unicode path bugs on Windows
    with open(image_path, "rb") as f:
        bytes_data = bytearray(f.read())
    numpy_array = np.asarray(bytes_data, dtype=np.uint8)
    image = cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(f"Could not decode image at path: {image_path}")

    return image

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Converts BGR image to single-channel grayscale."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def match_dimensions(
    img_reference: np.ndarray,
    img_target: np.ndarray,
    interpolation: int = cv2.INTER_AREA,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Resizes target image to match reference image dimensions (height, width).
    Returns (img_reference, resized_target).
    """
    h_ref, w_ref = img_reference.shape[:2]
    h_tgt, w_tgt = img_target.shape[:2]

    if (h_ref, w_ref) != (h_tgt, w_tgt):
        img_target = cv2.resize(img_target, (w_ref, h_ref), interpolation=interpolation)

    return img_reference, img_target

def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """Applies subtle Gaussian smoothing to reduce sensor noise while preserving edges."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
