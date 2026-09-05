"""
Unit and Integration Tests for ORBIT GEOINT PDF Report Generation.
"""

import os
import io
import json
import pytest
import numpy as np
import cv2
from PIL import Image

from app import create_app
from app.core.report_generator import generate_pdf_report


@pytest.fixture
def app_instance(tmp_path):
    """Create Flask test application instance with a temporary upload folder."""
    upload_dir = tmp_path / "test_uploads"
    upload_dir.mkdir(exist_ok=True)

    app = create_app({
        "TESTING": True,
        "UPLOAD_FOLDER": str(upload_dir),
        "MIN_DIMENSION": 64,
        "MAX_DIMENSION": 2048,
    })
    return app


@pytest.fixture
def client(app_instance):
    """Create Flask test client."""
    with app_instance.test_client() as client:
        yield client


@pytest.fixture
def mock_session_dir(tmp_path):
    """Creates a realistic mocked session directory with synthetic imagery and analysis metadata."""
    session_id = "orb-testpdf01"
    s_dir = tmp_path / session_id
    s_dir.mkdir()

    # Generate synthetic 256x256 test frames
    img_ref = np.full((256, 256, 3), 128, dtype=np.uint8)
    img_tgt = img_ref.copy()
    cv2.rectangle(img_tgt, (50, 50), (120, 120), (255, 255, 255), -1)
    
    img_overlay = img_tgt.copy()
    cv2.rectangle(img_overlay, (50, 50), (120, 120), (0, 0, 255), 2)
    
    img_heatmap = np.zeros((256, 256, 3), dtype=np.uint8)
    img_heatmap[50:120, 50:120] = [0, 0, 255]

    cv2.imwrite(str(s_dir / "before.png"), img_ref)
    cv2.imwrite(str(s_dir / "after.png"), img_tgt)
    cv2.imwrite(str(s_dir / "diff_overlay.png"), img_overlay)
    cv2.imwrite(str(s_dir / "diff_heatmap.png"), img_heatmap)

    # Save mock analysis results
    results_data = {
        "success": True,
        "session_id": session_id,
        "pipeline": "CLASSICAL_ORB_SSIM_V1",
        "model_type": "classical",
        "fallback_applied": False,
        "alignment": {
            "aligned": True,
            "status": "LOCKED",
            "inliers_count": 84,
            "inliers_ratio": 0.76,
        },
        "detection": {
            "severity": "HIGH",
            "primary_driver": "New Structural Construction",
            "headline": "Major Structural Transformation Detected (+7.5%)",
            "executive_summary": "Bi-temporal analysis detected 1 significant cluster of new physical structures.",
            "change_percentage": 7.45,
            "total_changed_pixels": 4900,
            "changed_regions_count": 1,
            "ssim_similarity_score": 0.9255,
            "overall_divergence": 0.0745,
            "sensitivity_applied": 0.35,
            "bounding_boxes": [
                {
                    "id": 1,
                    "label": "Zone #1 (Main Facility)",
                    "x": 50,
                    "y": 50,
                    "width": 70,
                    "height": 70,
                    "area_px": 4900,
                    "share_pct": 100.0,
                    "tag": "Industrial / Commercial Building",
                    "confidence": "High Confidence",
                }
            ],
        },
        "created_at": "2026-09-05 12:00:00 UTC",
    }

    with open(s_dir / "analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(results_data, f)

    return str(s_dir), session_id, results_data


def test_generate_pdf_report_direct(mock_session_dir):
    """Verifies that the ReportLab engine compiles a valid PDF document."""
    s_dir, session_id, results_data = mock_session_dir

    pdf_path = generate_pdf_report(session_dir=s_dir, session_id=session_id, results_data=results_data)

    assert os.path.exists(pdf_path), "PDF file was not created"
    assert os.path.getsize(pdf_path) > 2000, "PDF file size is suspiciously small"

    # Verify standard PDF magic header
    with open(pdf_path, "rb") as f:
        header = f.read(5)
        assert header == b"%PDF-", f"Expected %PDF- magic bytes, got {header}"


def test_download_pdf_route_not_found(client):
    """Verifies 404 response when requesting a report for a non-existent session."""
    res = client.get("/api/report/orb-nonexistent999/pdf")
    assert res.status_code == 404
    data = res.get_json()
    assert data["success"] is False
    assert "not found" in data["error"].lower()


def test_download_pdf_route_success(client, mock_session_dir, app_instance):
    """Verifies HTTP 200 and valid PDF payload delivery via the Flask endpoint."""
    s_dir, session_id, _ = mock_session_dir

    # Point app's UPLOAD_FOLDER to the temp directory parent
    app_instance.config["UPLOAD_FOLDER"] = os.path.dirname(s_dir)

    res = client.get(f"/api/report/{session_id}/pdf")

    assert res.status_code == 200
    assert "application/pdf" in res.headers.get("Content-Type", "")
    assert f"ORBIT_GEOINT_Report_{session_id}.pdf" in res.headers.get("Content-Disposition", "")
    assert res.data.startswith(b"%PDF-")
