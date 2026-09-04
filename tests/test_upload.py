"""
Automated Pytest Suite for ORBIT Image Ingestion & Validation Pipeline.
"""

import io
import os
import pytest
from PIL import Image, ImageDraw

from app import create_app

@pytest.fixture
def client(tmp_path):
    """Create Flask test client with a temporary uploads folder."""
    upload_dir = tmp_path / "test_uploads"
    upload_dir.mkdir()

    app = create_app({
        "TESTING": True,
        "UPLOAD_FOLDER": str(upload_dir),
        "MIN_DIMENSION": 64,  # Lower minimum for quick testing
        "MAX_DIMENSION": 2048,
    })

    with app.test_client() as client:
        yield client

def create_synthetic_image(width=256, height=256, color=(30, 80, 150), fmt="PNG"):
    """Helper to generate an in-memory test image with synthetic shapes."""
    img = Image.new("RGB", (width, height), color=color)
    draw = ImageDraw.Draw(img)
    draw.rectangle([20, 20, width - 20, height - 20], outline=(255, 255, 255), width=2)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf

def test_health_check(client):
    """Ensure telemetry health check reports online."""
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data["status"] == "ONLINE"
    assert data["system"] == "ORBIT_CORE_ENGINE"

def test_index_route(client):
    """Ensure platform template renders correctly."""
    res = client.get("/")
    assert res.status_code == 200
    assert b"ORBIT" in res.data
    assert b"Reference Baseline" in res.data or b"T-0" in res.data
    assert b"Target Observation" in res.data or b"T-1" in res.data

def test_upload_missing_payload(client):
    """Ensure 400 Bad Request if images are omitted."""
    res = client.post("/api/upload", data={}, content_type="multipart/form-data")
    assert res.status_code == 400
    data = res.get_json()
    assert data["success"] is False
    assert "Missing payload" in data["error"]

def test_upload_invalid_extension(client):
    """Ensure files with disallowed extensions are rejected."""
    bad_file = (io.BytesIO(b"fake data"), "script.exe")
    good_img = (create_synthetic_image(), "target.png")

    res = client.post(
        "/api/upload",
        data={"image_before": bad_file, "image_after": good_img},
        content_type="multipart/form-data",
    )
    assert res.status_code == 400
    data = res.get_json()
    assert data["success"] is False
    assert "details" in data
    assert "image_before" in data["details"]

def test_upload_corrupt_image(client):
    """Ensure corrupted image bytes fail PIL verification."""
    corrupt_file = (io.BytesIO(b"Not an actual image payload"), "broken.png")
    good_img = (create_synthetic_image(), "target.png")

    res = client.post(
        "/api/upload",
        data={"image_before": corrupt_file, "image_after": good_img},
        content_type="multipart/form-data",
    )
    assert res.status_code == 400
    data = res.get_json()
    assert data["success"] is False
    assert "Corrupted image or invalid header" in data["details"]["image_before"]

def test_upload_valid_pair(client):
    """Ensure valid image pairs are ingested, sessionized, and indexed."""
    img_before = (create_synthetic_image(256, 256, color=(40, 40, 40)), "sat_t0.png")
    img_after = (create_synthetic_image(256, 256, color=(80, 120, 200)), "sat_t1.png")

    res = client.post(
        "/api/upload",
        data={"image_before": img_before, "image_after": img_after},
        content_type="multipart/form-data",
    )
    assert res.status_code == 201
    data = res.get_json()
    assert data["success"] is True
    assert "orb-" in data["session_id"]
    assert data["status"] == "STAGED_FOR_STAGE_2"

    telemetry = data["telemetry"]
    assert telemetry["image_before"]["width"] == 256
    assert telemetry["image_before"]["height"] == 256
    assert telemetry["image_after"]["width"] == 256
    assert telemetry["image_after"]["height"] == 256
