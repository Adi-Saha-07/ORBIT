"""
Geospatial Evaluation Metrics Module for Satellite Change Detection.

Calculates Intersection over Union (IoU / Jaccard Index), F1-Score (Dice Coefficient),
Precision, Recall, and Overall Accuracy between predicted change masks and ground-truth
or comparative baseline masks.
"""

import numpy as np

def compute_change_metrics(
    pred_mask: np.ndarray,
    target_mask: np.ndarray,
    threshold: int = 127,
    eps: float = 1e-7,
) -> dict:
    """
    Computes standard semantic segmentation metrics for binary change detection.

    Args:
        pred_mask: Binary or uint8 prediction mask [0 or 255].
        target_mask: Binary or uint8 ground-truth / reference mask [0 or 255].
        threshold: Binarization cutoff if masks are continuous probabilities.
        eps: Small epsilon to prevent division by zero.

    Returns:
        dict containing:
            - iou (Jaccard Index)
            - f1_score (Dice Coefficient)
            - precision
            - recall
            - overall_accuracy
            - true_positives
            - false_positives
            - false_negatives
            - true_negatives
    """
    # Ensure boolean arrays
    p = (pred_mask > threshold).astype(bool)
    t = (target_mask > threshold).astype(bool)

    # Confusion Matrix Primitives
    tp = np.logical_and(p, t).sum()
    fp = np.logical_and(p, np.logical_not(t)).sum()
    fn = np.logical_and(np.logical_not(p), t).sum()
    tn = np.logical_and(np.logical_not(p), np.logical_not(t)).sum()

    total_pixels = tp + fp + fn + tn

    # Metrics
    intersection = tp
    union = tp + fp + fn

    iou = (intersection + eps) / (union + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    f1 = (2.0 * precision * recall + eps) / (precision + recall + eps)
    accuracy = (tp + tn) / (total_pixels + eps)

    return {
        "iou": round(float(iou), 4),
        "f1_score": round(float(f1), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "overall_accuracy": round(float(accuracy), 4),
        "confusion_matrix": {
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
        },
    }
