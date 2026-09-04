"""
Production Deep Learning Inference Wrapper for ORBIT.

Loads the trained Siamese U-Net (FC-Siam-diff) PyTorch model, manages tensor
preprocessing, and emits probability heatmaps and bounding contours compatible
with the Stage 2/3 visualizer.
"""

import os
import cv2
import numpy as np
import torch

from ml.model import SiameseUNetDiff
from app.core.preprocessor import match_dimensions

# Module-level model cache to avoid re-instantiating weights on every request
_CACHED_MODEL = None
_CACHED_DEVICE = None

def get_model(checkpoint_path: str = None) -> tuple[torch.nn.Module, torch.device]:
    """Retrieves or instantiates cached Siamese U-Net model."""
    global _CACHED_MODEL, _CACHED_DEVICE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if _CACHED_MODEL is not None and _CACHED_DEVICE == device:
        return _CACHED_MODEL, device

    model = SiameseUNetDiff(in_channels=3, out_channels=1, base_channels=32)

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict)
        print(f"[ORBIT-INFERENCE] Loaded weights from: {checkpoint_path}")
    else:
        # Default checkpoints lookup
        default_chk = os.path.join(
            os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
            "checkpoints",
            "best_model.pth"
        )
        if os.path.exists(default_chk):
            checkpoint = torch.load(default_chk, map_location=device)
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            model.load_state_dict(state_dict)
            print(f"[ORBIT-INFERENCE] Loaded default weights from: {default_chk}")

    model.to(device)
    model.eval()

    _CACHED_MODEL = model
    _CACHED_DEVICE = device
    return model, device

def preprocess_numpy_to_tensor(img_bgr: np.ndarray, target_size: tuple[int, int] = (256, 256)) -> torch.Tensor:
    """Preprocesses a single OpenCV BGR image into a normalized PyTorch tensor."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)

    arr = img_resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    norm = (arr - mean) / std

    tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor

def predict_change_siamese(
    img_reference: np.ndarray,
    img_target: np.ndarray,
    checkpoint_path: str = None,
    threshold: float = 0.5,
    min_contour_area: int = 150,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Executes Deep Learning Change Segmentation using Siamese U-Net.

    Returns:
        tuple (diff_intensity: np.ndarray, binary_mask: np.ndarray, metrics: dict)
    """
    img_reference, img_target = match_dimensions(img_reference, img_target)
    orig_h, orig_w = img_reference.shape[:2]
    total_pixels = orig_h * orig_w

    model, device = get_model(checkpoint_path)

    # 1. Preprocess tensors
    tensor_t0 = preprocess_numpy_to_tensor(img_reference, (256, 256)).to(device)
    tensor_t1 = preprocess_numpy_to_tensor(img_target, (256, 256)).to(device)

    # 2. Forward inference pass
    with torch.no_grad():
        logits = model(tensor_t0, tensor_t1)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    # 3. Resize probability map back to original spatial canvas
    prob_map_orig = cv2.resize(probs, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    diff_intensity = np.clip(prob_map_orig * 255.0, 0, 255).astype(np.uint8)

    # 4. Threshold into binary change mask
    thresh_val = int(threshold * 255)
    _, binary_mask = cv2.threshold(diff_intensity, thresh_val, 255, cv2.THRESH_BINARY)

    # 5. Morphological smoothing to join adjacent pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)

    # 6. Extract Contours & Bounding Reticles
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    filtered_mask = np.zeros_like(binary_mask)
    bounding_boxes = []

    for idx, cnt in enumerate(contours):
        area = float(cv2.contourArea(cnt))
        if area < min_contour_area:
            continue

        cv2.drawContours(filtered_mask, [cnt], -1, 255, thickness=cv2.FILLED)
        bx, by, bw, bh = cv2.boundingRect(cnt)

        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = bx + bw // 2, by + bh // 2

        bounding_boxes.append({
            "id": idx + 1,
            "x": int(bx),
            "y": int(by),
            "width": int(bw),
            "height": int(bh),
            "area_px": int(area),
            "centroid": [cx, cy],
        })

    bounding_boxes.sort(key=lambda b: b["area_px"], reverse=True)

    changed_pixels = int(np.count_nonzero(filtered_mask))
    change_pct = round((changed_pixels / total_pixels) * 100.0, 3)
    mean_confidence = round(float(np.mean(prob_map_orig)), 4)

    from app.core.detector import summarize_findings
    summary_data = summarize_findings(change_pct, len(bounding_boxes), bounding_boxes, total_pixels)

    metrics = {
        "model_architecture": "SiameseUNetDiff_v1",
        "mean_change_confidence": mean_confidence,
        "change_percentage": change_pct,
        "total_changed_pixels": changed_pixels,
        "total_image_pixels": total_pixels,
        "changed_regions_count": len(bounding_boxes),
        "bounding_boxes": bounding_boxes,
        "threshold_applied": threshold,
        "device": str(device),
        "executive_summary": summary_data["executive_summary"],
        "headline": summary_data["headline"],
        "severity": summary_data["severity"],
        "primary_driver": summary_data["primary_driver"],
    }

    return diff_intensity, filtered_mask, metrics

