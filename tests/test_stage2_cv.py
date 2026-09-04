"""
Automated Pytest Suite for Stage 2: Classical CV (ORB Alignment + SSIM Diff Engine).
"""

import io
import os
import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw

from app import create_app
from app.core.aligner import align_images_orb
from app.core.detector import detect_changes_ssim
from app.core.visualizer import generate_diff_heatmap, create_diff_overlay

@pytest.fixture
def client(tmp_path):
    """Create Flask test client with isolated temp directory."""
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

def generate_satellite_mock(width=300, height=300, add_construction=False, rotate_deg=0):
    """
    Generates a realistic synthetic aerial image with texture, roads, and land plots.
    If add_construction is True, adds a new industrial complex in the center-right.
    """
    # Background terrain (earth tones)
    img = np.full((height, width, 3), (35, 75, 45), dtype=np.uint8)

    # Add textured noise to simulate vegetation
    noise = np.random.RandomState(42).randint(-15, 15, (height, width, 3))
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Draw asphalt roads
    cv2.line(img, (0, 100), (width, 100), (90, 90, 95), 8)
    cv2.line(img, (150, 0), (150, height), (90, 90, 95), 8)

    # Existing structures in T-0 baseline
    cv2.rectangle(img, (30, 30), (80, 80), (160, 150, 140), -1)
    cv2.rectangle(img, (30, 140), (90, 220), (140, 130, 120), -1)

    # If T-1 capture has new construction
    if add_construction:
        # New bright warehouse + parking clearing
        cv2.rectangle(img, (180, 130), (270, 230), (220, 225, 230), -1)
        cv2.rectangle(img, (190, 140), (230, 180), (70, 70, 190), -1)

    # Optional slight rotation/shift to test homography recovery
    if rotate_deg != 0:
        center = (width // 2, height // 2)
        M = cv2.getRotationMatrix2D(center, rotate_deg, 1.0)
        img = cv2.warpAffine(img, M, (width, height), borderMode=cv2.BORDER_REFLECT)

    return img

def np_to_bytes(img_bgr, fmt="PNG"):
    """Encodes numpy BGR image to BytesIO for test uploads."""
    is_success, buffer = cv2.imencode(".png" if fmt == "PNG" else ".jpg", img_bgr)
    return io.BytesIO(buffer)

def test_orb_alignment_identical():
    """Ensure ORB alignment recognizes identical frames with locked status."""
    img = generate_satellite_mock(300, 300)
    aligned, H, telemetry = align_images_orb(img, img)

    assert telemetry["aligned"] is True
    assert telemetry["status"] == "ALIGNMENT_LOCKED"
    assert telemetry["inliers"] > 20
    assert H is not None

def test_orb_alignment_recovers_rotation():
    """Ensure ORB alignment recovers perspective when frame is rotated slightly."""
    img_ref = generate_satellite_mock(300, 300)
    img_tgt = generate_satellite_mock(300, 300, rotate_deg=4)

    aligned, H, telemetry = align_images_orb(img_ref, img_tgt)

    assert telemetry["aligned"] is True
    assert telemetry["inliers"] >= 10
    assert H is not None
    # Aligned image should now be closer to reference than unaligned target
    diff_before_align = np.mean(np.abs(img_ref.astype(float) - img_tgt.astype(float)))
    diff_after_align = np.mean(np.abs(img_ref.astype(float) - aligned.astype(float)))
    assert diff_after_align < diff_before_align

def test_ssim_detection_identical():
    """Ensure SSIM reports ~0% change for identical imagery."""
    img = generate_satellite_mock(300, 300)
    diff_intensity, binary_mask, metrics = detect_changes_ssim(
        img_reference=img,
        img_target=img,
        sensitivity=0.35,
    )

    assert metrics["ssim_similarity_score"] >= 0.99
    assert metrics["change_percentage"] == 0.0
    assert metrics["changed_regions_count"] == 0
    assert len(metrics["bounding_boxes"]) == 0

def test_ssim_detection_synthetic_construction():
    """Ensure SSIM and contour engine detect synthetic industrial building addition."""
    img_ref = generate_satellite_mock(300, 300, add_construction=False)
    img_tgt = generate_satellite_mock(300, 300, add_construction=True)

    diff_intensity, binary_mask, metrics = detect_changes_ssim(
        img_reference=img_ref,
        img_target=img_tgt,
        sensitivity=0.30,
        min_contour_area=150,
    )

    assert metrics["ssim_similarity_score"] < 0.95
    assert metrics["change_percentage"] > 2.0  # Significant change
    assert metrics["changed_regions_count"] >= 1

    # Verify bounding box overlays the construction region (around x=180-270, y=130-230)
    main_box = metrics["bounding_boxes"][0]
    assert main_box["x"] >= 150
    assert main_box["y"] >= 110
    assert main_box["area_px"] > 500

def test_visualizer_outputs():
    """Ensure heatmap and diff overlay generate correct dimensions and channel count."""
    img_tgt = generate_satellite_mock(300, 300, add_construction=True)
    diff_int = np.full((300, 300), 128, dtype=np.uint8)
    mask = np.zeros((300, 300), dtype=np.uint8)
    mask[100:200, 100:200] = 255
    boxes = [{"id": 1, "x": 100, "y": 100, "width": 100, "height": 100, "area_px": 10000}]

    heatmap = generate_diff_heatmap(diff_int)
    assert heatmap.shape == (300, 300, 3)

    overlay = create_diff_overlay(img_tgt, diff_int, mask, boxes)
    assert overlay.shape == (300, 300, 3)

def test_api_analyze_end_to_end(client):
    """End-to-end integration test: Ingest dual frames, trigger analysis, verify telemetry and artifacts."""
    img_t0 = generate_satellite_mock(300, 300, add_construction=False)
    img_t1 = generate_satellite_mock(300, 300, add_construction=True)

    # 1. Ingest images via /api/upload
    upload_res = client.post(
        "/api/upload",
        data={
            "image_before": (np_to_bytes(img_t0), "sat_t0.png"),
            "image_after": (np_to_bytes(img_t1), "sat_t1.png"),
        },
        content_type="multipart/form-data",
    )
    assert upload_res.status_code == 201
    upload_data = upload_res.get_json()
    session_id = upload_data["session_id"]

    # 2. Trigger analysis via /api/analyze
    analyze_res = client.post(
        "/api/analyze",
        json={
            "session_id": session_id,
            "sensitivity": 0.30,
            "min_contour_area": 100,
            "enable_alignment": True,
        },
    )
    assert analyze_res.status_code == 200
    res_data = analyze_res.get_json()

    assert res_data["success"] is True
    assert res_data["session_id"] == session_id
    assert "alignment" in res_data
    assert "detection" in res_data
    assert "artifacts" in res_data

    # Check metrics
    detection = res_data["detection"]
    assert detection["change_percentage"] > 1.0
    assert detection["changed_regions_count"] >= 1

    # Check artifact URLs
    artifacts = res_data["artifacts"]
    assert artifacts["heatmap_url"].endswith("diff_heatmap.png")
    assert artifacts["overlay_url"].endswith("diff_overlay.png")

    # 3. Verify serving artifacts
    artifact_res = client.get(artifacts["overlay_url"])
    assert artifact_res.status_code == 200
    assert len(artifact_res.data) > 0
