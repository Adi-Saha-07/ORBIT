"""
Benchmark and Evaluation Script for ORBIT.

Compares Classical SSIM vs Siamese U-Net (FC-Siam-diff) across:
- IoU (Intersection over Union)
- F1-Score (Dice)
- Precision & Recall
- Inference Latency (ms)

Generates clean Markdown benchmark tables ready for portfolio and resume inclusion.
"""

import time
import os
import sys
import numpy as np
import cv2

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from app.core.detector import detect_changes_ssim
from app.core.ml_inference import predict_change_siamese
from app.core.metrics import compute_change_metrics

def generate_benchmark_pair(size=(512, 512)):
    """Generates synthetic high-res satellite pair with known ground-truth mask."""
    w, h = size
    t0 = np.full((h, w, 3), (40, 80, 50), dtype=np.uint8)
    t1 = t0.copy()

    # Roads
    cv2.line(t0, (0, 250), (w, 250), (90, 90, 95), 10)
    cv2.line(t1, (0, 250), (w, 250), (90, 90, 95), 10)

    # Ground truth change mask
    gt_mask = np.zeros((h, w), dtype=np.uint8)

    # 1. New residential block
    cv2.rectangle(t1, (100, 100), (220, 200), (210, 210, 215), -1)
    cv2.rectangle(gt_mask, (100, 100), (220, 200), 255, -1)

    # 2. Deforestation / cleared soil plot
    cv2.rectangle(t1, (300, 300), (450, 420), (140, 90, 50), -1)
    cv2.rectangle(gt_mask, (300, 300), (450, 420), 255, -1)

    return t0, t1, gt_mask

def run_benchmark():
    print("\n" + "=" * 75)
    print("  ORBIT // SATELLITE CHANGE DETECTION PIPELINE BENCHMARK")
    print("=" * 75)

    img_t0, img_t1, gt_mask = generate_benchmark_pair()

    # 1. Benchmark Classical SSIM Pipeline
    start_time = time.perf_counter()
    _, ssim_mask, ssim_metrics = detect_changes_ssim(img_t0, img_t1, sensitivity=0.32)
    ssim_latency = (time.perf_counter() - start_time) * 1000

    ssim_eval = compute_change_metrics(ssim_mask, gt_mask)

    # 2. Benchmark Deep Learning Siamese U-Net Pipeline
    start_time = time.perf_counter()
    _, ml_mask, ml_metrics = predict_change_siamese(img_t0, img_t1, threshold=0.5)
    ml_latency = (time.perf_counter() - start_time) * 1000

    ml_eval = compute_change_metrics(ml_mask, gt_mask)

    # Comparative Metric Printout
    print("\n### BENCHMARK EVALUATION RESULTS (Ready for Portfolio / Resume)\n")
    print(f"| Pipeline Architecture | IoU (Jaccard) | F1-Score (Dice) | Precision | Recall | Latency (CPU) |")
    print(f"| :--- | :---: | :---: | :---: | :---: | :---: |")
    print(f"| **Classical SSIM (MVP)** | {ssim_eval['iou']:.4f} | {ssim_eval['f1_score']:.4f} | {ssim_eval['precision']:.4f} | {ssim_eval['recall']:.4f} | {ssim_latency:.1f} ms |")
    print(f"| **Siamese U-Net (Deep Learning)** | {ml_eval['iou']:.4f} | {ml_eval['f1_score']:.4f} | {ml_eval['precision']:.4f} | {ml_eval['recall']:.4f} | {ml_latency:.1f} ms |")
    print("\n" + "=" * 75 + "\n")

    return {
        "classical": {**ssim_eval, "latency_ms": ssim_latency},
        "deep_learning": {**ml_eval, "latency_ms": ml_latency},
    }

if __name__ == "__main__":
    run_benchmark()
