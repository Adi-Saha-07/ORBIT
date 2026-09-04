"""
Visualization and Visual Diff Composite Module for ORBIT (Stage 2 & Stage 3).

Generates high-contrast thermal diff heatmaps, transparent HUD overlays,
and bounding reticles with telemetry markers.
"""

import os
import cv2
import numpy as np

def save_image_bgr(image: np.ndarray, output_path: str) -> bool:
    """Safely saves a BGR image to disk using imencode (Unicode safe)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    ext = os.path.splitext(output_path)[1].lower() or ".png"
    success, encoded = cv2.imencode(ext, image)
    if not success:
        return False
    with open(output_path, "wb") as f:
        f.write(encoded.tobytes())
    return True

def generate_diff_heatmap(
    diff_intensity: np.ndarray,
    colormap: int = cv2.COLORMAP_TURBO,
) -> np.ndarray:
    """
    Transforms single-channel difference intensity into a vibrant thermal/spectral heatmap.
    Uses cv2.COLORMAP_TURBO for smooth, high-dynamic-range color transitions.
    """
    if len(diff_intensity.shape) == 3:
        diff_intensity = cv2.cvtColor(diff_intensity, cv2.COLOR_BGR2GRAY)

    heatmap = cv2.applyColorMap(diff_intensity, colormap)
    return heatmap

def create_diff_overlay(
    target_image: np.ndarray,
    diff_intensity: np.ndarray,
    binary_mask: np.ndarray,
    bounding_boxes: list[dict] = None,
    alpha: float = 0.55,
    draw_reticles: bool = True,
) -> np.ndarray:
    """
    Composites the change heatmap over the target satellite frame with cyberpunk HUD bounding reticles.

    Args:
        target_image: Aligned target image (T-1) in BGR.
        diff_intensity: Normalized difference intensity map (0-255).
        binary_mask: Cleaned binary mask of confirmed changes.
        bounding_boxes: List of bounding box telemetry dicts [{'x', 'y', 'width', 'height', 'id'}].
        alpha: Blend weight of the heatmap [0.0 = only target, 1.0 = only heatmap].
        draw_reticles: Whether to annotate HUD corner brackets and ID badges.

    Returns:
        np.ndarray: Blended BGR image ready for rendering or export.
    """
    # 1. Generate full heatmap
    heatmap = generate_diff_heatmap(diff_intensity, colormap=cv2.COLORMAP_TURBO)

    # 2. Blend heatmap only where binary_mask indicates significant change
    overlay = target_image.copy()

    # Create 3-channel boolean mask
    mask_3c = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR) > 0

    # Alpha composite on changed pixels
    blended_region = cv2.addWeighted(target_image, 1.0 - alpha, heatmap, alpha, 0)
    overlay[mask_3c] = blended_region[mask_3c]

    # Add subtle tinted contour border around changed zones (Cyber Neon Teal/Amber)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 240, 0), 1)  # BGR Cyan contour

    # 3. Draw Cyberpunk HUD Reticles around Bounding Boxes
    if draw_reticles and bounding_boxes:
        for box in bounding_boxes:
            bx, by, bw, bh = box["x"], box["y"], box["width"], box["height"]
            box_id = box.get("id", 1)
            area = box.get("area_px", 0)

            # Neon Amber BGR: (0, 170, 255)
            color_neon = (0, 170, 255)
            line_len = min(16, bw // 3, bh // 3)

            # Draw HUD corner brackets instead of plain rectangles
            # Top-Left corner
            cv2.line(overlay, (bx, by), (bx + line_len, by), color_neon, 2)
            cv2.line(overlay, (bx, by), (bx, by + line_len), color_neon, 2)

            # Top-Right corner
            cv2.line(overlay, (bx + bw, by), (bx + bw - line_len, by), color_neon, 2)
            cv2.line(overlay, (bx + bw, by), (bx + bw, by + line_len), color_neon, 2)

            # Bottom-Left corner
            cv2.line(overlay, (bx, by + bh), (bx + line_len, by + bh), color_neon, 2)
            cv2.line(overlay, (bx, by + bh), (bx, by + bh - line_len), color_neon, 2)

            # Bottom-Right corner
            cv2.line(overlay, (bx + bw, by + bh), (bx + bw - line_len, by + bh), color_neon, 2)
            cv2.line(overlay, (bx + bw, by + bh), (bx + bw, by + bh - line_len), color_neon, 2)

            # Telemetry label badge above box
            label = f"DELTA #{box_id}: {area}px"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.38
            thickness = 1
            (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Background pill for readability
            label_y = max(by - 5, th + 5)
            cv2.rectangle(
                overlay,
                (bx, label_y - th - 3),
                (bx + tw + 6, label_y + baseline),
                (10, 15, 25),
                cv2.FILLED,
            )
            cv2.rectangle(
                overlay,
                (bx, label_y - th - 3),
                (bx + tw + 6, label_y + baseline),
                color_neon,
                1,
            )
            cv2.putText(
                overlay,
                label,
                (bx + 3, label_y - 2),
                font,
                font_scale,
                (0, 240, 255),
                thickness,
                cv2.LINE_AA,
            )

    return overlay
