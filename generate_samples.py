"""
Sample Satellite Imagery Generator for ORBIT.
Generates a realistic paired temporal scenario:
- T-0 (Baseline): Rural farmland, winding river, highway intersection.
- T-1 (Target): A new solar farm / industrial warehouse complex built in the agricultural plot,
  plus a slight 2.5-degree perspective tilt to simulate orbital satellite drift.
"""

import os
import cv2
import numpy as np

def generate_orbital_samples():
    output_dir = os.path.join(os.path.dirname(__file__), "samples")
    os.makedirs(output_dir, exist_ok=True)

    w, h = 640, 640
    # 1. Base terrain: green-brown agricultural landscape
    img_t0 = np.full((h, w, 3), (38, 78, 48), dtype=np.uint8)

    # Add realistic texture noise
    rng = np.random.RandomState(1337)
    noise = rng.randint(-18, 18, (h, w, 3))
    img_t0 = np.clip(img_t0.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Crop field divisions (patches of varying soil tones)
    cv2.rectangle(img_t0, (40, 40), (280, 260), (45, 95, 60), -1)
    cv2.rectangle(img_t0, (300, 40), (600, 260), (35, 70, 45), -1)
    cv2.rectangle(img_t0, (40, 320), (280, 600), (30, 85, 55), -1)
    cv2.rectangle(img_t0, (320, 320), (600, 600), (42, 80, 50), -1)

    # Winding river (deep blue-cyan water)
    pts = np.array([
        [0, 180], [120, 190], [240, 220], [350, 210],
        [460, 240], [550, 230], [640, 260],
        [640, 295], [550, 265], [460, 275],
        [350, 245], [240, 255], [120, 225], [0, 215]
    ], np.int32)
    cv2.fillPoly(img_t0, [pts], (140, 80, 30))

    # Highway intersection (grey asphalt)
    cv2.line(img_t0, (300, 0), (300, h), (85, 88, 92), 14)
    cv2.line(img_t0, (0, 450), (w, 450), (85, 88, 92), 14)
    # Highway markings
    cv2.line(img_t0, (300, 0), (300, h), (230, 230, 230), 2)
    cv2.line(img_t0, (0, 450), (w, 450), (230, 230, 230), 2)

    # Small existing farmhouse complex
    cv2.rectangle(img_t0, (100, 80), (160, 140), (180, 175, 170), -1)
    cv2.rectangle(img_t0, (120, 95), (145, 125), (60, 60, 160), -1)

    # Save T-0 Baseline
    path_t0 = os.path.join(output_dir, "satellite_t0_baseline.png")
    cv2.imwrite(path_t0, img_t0)

    # 2. T-1 Target (Temporal Change + Slight Orbital Tilt)
    img_t1 = img_t0.copy()

    # Major Change 1: New Solar Farm / Industrial Complex in bottom-right field
    cv2.rectangle(img_t1, (360, 360), (560, 560), (210, 215, 220), -1)  # Cleared gravel plot
    # Solar panel rows (deep blue reflective rectangles)
    for y in range(380, 540, 28):
        cv2.rectangle(img_t1, (380, y), (540, y + 16), (180, 90, 20), -1)

    # Major Change 2: Commercial logistics facility in top-left field
    cv2.rectangle(img_t1, (80, 360), (240, 420), (230, 230, 235), -1)  # Warehouse roof
    cv2.rectangle(img_t1, (90, 370), (140, 410), (120, 120, 130), -1)

    # Simulate Orbital Sensor Drift: 2.0-degree rotation + 8px translation
    center = (w // 2, h // 2)
    M_rot = cv2.getRotationMatrix2D(center, 2.0, 1.0)
    M_rot[0, 2] += 8.0
    M_rot[1, 2] -= 6.0
    img_t1_drifted = cv2.warpAffine(img_t1, M_rot, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Save T-1 Target
    path_t1 = os.path.join(output_dir, "satellite_t1_target.png")
    cv2.imwrite(path_t1, img_t1_drifted)

    print(f"[SAMPLE-GEN] Created: {path_t0}")
    print(f"[SAMPLE-GEN] Created: {path_t1}")

if __name__ == "__main__":
    generate_orbital_samples()
