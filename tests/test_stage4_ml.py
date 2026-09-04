"""
Automated Pytest Suite for Stage 4: Deep Learning Pipeline (Siamese U-Net & Metrics).
"""

import io
import os
import cv2
import numpy as np
import pytest
import torch

from app import create_app
from ml.model import SiameseUNetDiff
from ml.loss import BCEDiceLoss, DiceLoss
from app.core.metrics import compute_change_metrics
from app.core.ml_inference import predict_change_siamese

@pytest.fixture
def client(tmp_path):
    upload_dir = tmp_path / "test_uploads"
    upload_dir.mkdir()

    app = create_app({
        "TESTING": True,
        "UPLOAD_FOLDER": str(upload_dir),
        "MIN_DIMENSION": 64,
        "MAX_DIMENSION": 2048,
    })

    with app.test_client() as client:
        yield client

def test_siamese_unet_forward_pass():
    """Verify Siamese U-Net takes twin (B, 3, H, W) images and outputs (B, 1, H, W) logits."""
    model = SiameseUNetDiff(in_channels=3, out_channels=1, base_channels=16)
    model.eval()

    batch_size = 2
    h, w = 128, 128
    t0 = torch.randn(batch_size, 3, h, w)
    t1 = torch.randn(batch_size, 3, h, w)

    with torch.no_grad():
        logits = model(t0, t1)

    assert logits.shape == (batch_size, 1, h, w)

def test_bce_dice_loss_and_gradients():
    """Verify hybrid loss computes valid scalar and enables backprop."""
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

    logits = torch.randn(2, 1, 64, 64, requires_grad=True)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()

    loss = criterion(logits, targets)
    assert not torch.isnan(loss)
    assert loss.item() > 0.0

    loss.backward()
    assert logits.grad is not None

def test_evaluation_metrics_math():
    """Verify IoU and F1 score mathematical correctness on synthetic masks."""
    # Create 100x100 ground truth with 400 positive pixels (20x20)
    gt = np.zeros((100, 100), dtype=np.uint8)
    gt[10:30, 10:30] = 255  # 400 px

    # Create prediction overlapping 200 pixels and having 200 false positives
    pred = np.zeros((100, 100), dtype=np.uint8)
    pred[10:30, 20:40] = 255  # 400 px total: 200 overlap, 200 FP, 200 FN

    metrics = compute_change_metrics(pred, gt)

    # TP = 200, FP = 200, FN = 200
    # IoU = TP / (TP + FP + FN) = 200 / 600 = 0.3333
    # Precision = 200 / 400 = 0.5
    # Recall = 200 / 400 = 0.5
    # F1 = 0.5000
    assert pytest.approx(metrics["iou"], 0.01) == 0.3333
    assert pytest.approx(metrics["f1_score"], 0.01) == 0.5000
    assert pytest.approx(metrics["precision"], 0.01) == 0.5000
    assert pytest.approx(metrics["recall"], 0.01) == 0.5000

def test_ml_inference_wrapper_execution():
    """Verify production inference wrapper produces valid maps and telemetry."""
    img_t0 = np.full((256, 256, 3), 50, dtype=np.uint8)
    img_t1 = img_t0.copy()
    img_t1[50:120, 50:120] = 220  # Synthetic construction

    diff_int, binary_mask, metrics = predict_change_siamese(img_t0, img_t1, threshold=0.4)

    assert diff_int.shape == (256, 256)
    assert binary_mask.shape == (256, 256)
    assert "model_architecture" in metrics
    assert "mean_change_confidence" in metrics

def test_api_swappable_siamese_pipeline(client):
    """Verify POST /api/analyze supports model_type='siamese_unet'."""
    img = np.full((200, 200, 3), 60, dtype=np.uint8)
    _, buf = cv2.imencode(".png", img)
    bytes_io = io.BytesIO(buf)

    # Ingest pair
    up_res = client.post(
        "/api/upload",
        data={"image_before": (bytes_io, "t0.png"), "image_after": (io.BytesIO(buf), "t1.png")},
        content_type="multipart/form-data"
    )
    assert up_res.status_code == 201
    session_id = up_res.get_json()["session_id"]

    # Trigger ML analysis
    an_res = client.post(
        "/api/analyze",
        json={"session_id": session_id, "model_type": "siamese_unet", "sensitivity": 0.4}
    )
    assert an_res.status_code == 200
    data = an_res.get_json()
    assert data["pipeline"] == "DEEP_LEARNING_SIAMESE_UNET_V2"
    assert data["model_type"] == "siamese_unet"
    assert "artifacts" in data

def test_api_benchmark_endpoint(client):
    """Verify POST /api/benchmark computes IoU & F1 comparison between Classical & ML."""
    img_t0 = np.full((200, 200, 3), 40, dtype=np.uint8)
    img_t1 = img_t0.copy()
    img_t1[50:100, 50:100] = 200

    _, b0 = cv2.imencode(".png", img_t0)
    _, b1 = cv2.imencode(".png", img_t1)

    up_res = client.post(
        "/api/upload",
        data={"image_before": (io.BytesIO(b0), "t0.png"), "image_after": (io.BytesIO(b1), "t1.png")},
        content_type="multipart/form-data"
    )
    session_id = up_res.get_json()["session_id"]

    # Call /api/benchmark
    bench_res = client.post("/api/benchmark", json={"session_id": session_id})
    assert bench_res.status_code == 200
    bdata = bench_res.get_json()

    assert bdata["success"] is True
    assert "classical" in bdata
    assert "deep_learning" in bdata
    assert "agreement_metrics" in bdata
    assert "iou_overlap" in bdata["agreement_metrics"]
    assert "f1_agreement" in bdata["agreement_metrics"]
