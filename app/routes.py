"""
HTTP Route Handlers for the ORBIT Platform.

Exposes endpoints for health monitoring, template rendering, and dual-frame
image ingestion with comprehensive validation.
"""

import os
import uuid
import json
import datetime
from flask import Blueprint, render_template, request, jsonify, current_app, send_from_directory, send_file
from werkzeug.utils import secure_filename

from app.core.validator import validate_image_pair
from app.core.preprocessor import load_image_bgr
from app.core.aligner import align_images_orb
from app.core.detector import detect_changes, detect_changes_ssim
from app.core.metrics import compute_change_metrics
from app.core.visualizer import generate_diff_heatmap, create_diff_overlay, save_image_bgr
from app.core.report_generator import generate_pdf_report

main_bp = Blueprint("main", __name__)

@main_bp.route("/")
def index():
    """Renders the ORBIT Cyberpunk HUD telemetry dashboard."""
    return render_template("index.html")

@main_bp.route("/api/health", methods=["GET"])
def health_check():
    """Telemetry health check endpoint."""
    return jsonify({
        "status": "ONLINE",
        "system": "ORBIT_CORE_ENGINE",
        "version": "1.0.0-stage1",
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "capabilities": [
            "DUAL_FRAME_VALIDATION",
            "PIL_HEADER_INTEGRITY",
            "SESSION_ISOLATION",
        ],
    }), 200

@main_bp.route("/api/upload", methods=["POST"])
def upload_pair():
    """
    Ingests and validates a pair of temporal satellite captures.

    Expected form-data:
      - image_before: FileStorage (T-0 Reference frame)
      - image_after: FileStorage (T-1 Target frame)

    Returns:
      JSON with session_id, validated telemetry metadata, and staging status.
    """
    if "image_before" not in request.files or "image_after" not in request.files:
        return jsonify({
            "success": False,
            "error": "Missing payload. Both 'image_before' and 'image_after' files are required.",
        }), 400

    file_before = request.files["image_before"]
    file_after = request.files["image_after"]

    # Retrieve configuration rules
    allowed_exts = current_app.config.get("ALLOWED_EXTENSIONS")
    min_dim = current_app.config.get("MIN_DIMENSION", 256)
    max_dim = current_app.config.get("MAX_DIMENSION", 4096)

    # Perform structural & header validation
    is_valid, errors, metadata = validate_image_pair(
        file_before=file_before,
        file_after=file_after,
        allowed_extensions=allowed_exts,
        min_dim=min_dim,
        max_dim=max_dim,
    )

    if not is_valid:
        return jsonify({
            "success": False,
            "error": "Validation failed on one or both orbital captures.",
            "details": errors,
        }), 400

    # Generate isolated session identifier
    session_id = f"orb-{uuid.uuid4().hex[:10]}"
    session_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], session_id)
    os.makedirs(session_dir, exist_ok=True)

    # Preserve file extensions for storage
    ext_before = secure_filename(file_before.filename).rsplit(".", 1)[-1].lower()
    ext_after = secure_filename(file_after.filename).rsplit(".", 1)[-1].lower()

    before_save_name = f"before.{ext_before}"
    after_save_name = f"after.{ext_after}"

    path_before = os.path.join(session_dir, before_save_name)
    path_after = os.path.join(session_dir, after_save_name)

    file_before.save(path_before)
    file_after.save(path_after)

    # Attach storage paths to metadata
    metadata["image_before"]["saved_as"] = before_save_name
    metadata["image_after"]["saved_as"] = after_save_name

    return jsonify({
        "success": True,
        "message": "Orbital frame pair successfully ingested, verified, and staged.",
        "session_id": session_id,
        "telemetry": metadata,
        "status": "STAGED_FOR_STAGE_2",
        "next_step": "FEATURE_ALIGNMENT_AND_SSIM_DIFF",
    }), 201

@main_bp.route("/api/analyze", methods=["POST"])
def analyze_session():
    """
    Executes Stage 2 Classical CV change detection pipeline on a staged session.

    Payload JSON:
      - session_id: str (Required)
      - sensitivity: float [0.1 to 0.9, default 0.35]
      - min_contour_area: int [pixels, default 150]
      - enable_alignment: bool [default True]

    Returns:
      JSON with alignment telemetry, SSIM change metrics, and URLs to visual artifacts.
    """
    payload = request.get_json(silent=True) or request.form.to_dict()
    session_id = payload.get("session_id")

    if not session_id:
        return jsonify({
            "success": False,
            "error": "Missing required parameter 'session_id'.",
        }), 400

    clean_session_id = secure_filename(session_id)
    session_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], clean_session_id)

    if not os.path.isdir(session_dir):
        return jsonify({
            "success": False,
            "error": f"Session '{session_id}' not found or expired.",
        }), 404

    # Locate before and after frames in the session directory
    files_in_session = os.listdir(session_dir)
    before_file = next((f for f in files_in_session if f.startswith("before.")), None)
    after_file = next((f for f in files_in_session if f.startswith("after.")), None)

    if not before_file or not after_file:
        return jsonify({
            "success": False,
            "error": "Session directory is missing reference ('before') or target ('after') frame.",
        }), 400

    path_before = os.path.join(session_dir, before_file)
    path_after = os.path.join(session_dir, after_file)

    try:
        img_reference = load_image_bgr(path_before)
        img_target = load_image_bgr(path_after)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to decode session imagery: {str(e)}",
        }), 500

    # Parse configurable parameters
    sensitivity = float(payload.get("sensitivity", 0.35))
    min_contour_area = int(payload.get("min_contour_area", 150))
    enable_alignment = str(payload.get("enable_alignment", "true")).lower() in {"true", "1", "yes"}
    model_type = str(payload.get("model_type", "classical")).lower()

    # 1. Feature Matching & Homography Alignment
    if enable_alignment:
        aligned_target, homography, align_telemetry = align_images_orb(
            img_reference=img_reference,
            img_target=img_target,
        )
    else:
        aligned_target = img_target
        align_telemetry = {
            "aligned": False,
            "status": "BYPASSED_BY_USER",
            "homography_matrix": None,
        }

    # Save aligned frame
    path_aligned = os.path.join(session_dir, "aligned.png")
    save_image_bgr(aligned_target, path_aligned)

    # 2. Change Detection (Swappable: Classical SSIM vs Siamese U-Net)
    diff_intensity, binary_mask, detection_metrics = detect_changes(
        img_reference=img_reference,
        img_target=aligned_target,
        model_type=model_type,
        sensitivity=sensitivity,
        min_contour_area=min_contour_area,
    )

    # 3. Visual Artifact Generation
    diff_heatmap = generate_diff_heatmap(diff_intensity)
    diff_overlay = create_diff_overlay(
        target_image=aligned_target,
        diff_intensity=diff_intensity,
        binary_mask=binary_mask,
        bounding_boxes=detection_metrics["bounding_boxes"],
    )

    path_heatmap = os.path.join(session_dir, "diff_heatmap.png")
    path_overlay = os.path.join(session_dir, "diff_overlay.png")

    save_image_bgr(diff_heatmap, path_heatmap)
    save_image_bgr(diff_overlay, path_overlay)

    is_fallback = bool(detection_metrics.get("fallback_applied", False))
    pipeline_name = "DEEP_LEARNING_SIAMESE_UNET_V2" if (model_type == "siamese_unet" and not is_fallback) else "CLASSICAL_ORB_SSIM_V1"

    response_payload = {
        "success": True,
        "session_id": session_id,
        "pipeline": pipeline_name,
        "model_type": "classical" if is_fallback else model_type,
        "fallback_applied": is_fallback,
        "alignment": align_telemetry,
        "detection": detection_metrics,
        "artifacts": {
            "reference_url": f"/uploads/{clean_session_id}/{before_file}",
            "target_url": f"/uploads/{clean_session_id}/{after_file}",
            "aligned_url": f"/uploads/{clean_session_id}/aligned.png",
            "heatmap_url": f"/uploads/{clean_session_id}/diff_heatmap.png",
            "overlay_url": f"/uploads/{clean_session_id}/diff_overlay.png",
            "report_url": f"/api/report/{clean_session_id}/pdf",
        },
        "created_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

    # Persist results metadata to disk so PDF report generator can retrieve full telemetry
    try:
        results_json_path = os.path.join(session_dir, "analysis_results.json")
        with open(results_json_path, "w", encoding="utf-8") as jf:
            json.dump(response_payload, jf, indent=2)
    except Exception as e:
        current_app.logger.warning(f"Could not cache analysis_results.json: {e}")

    return jsonify(response_payload), 200

@main_bp.route("/api/benchmark", methods=["POST"])
def benchmark_session():
    """
    Executes comparative benchmark between Classical SSIM and Siamese U-Net ML.
    Computes cross-model IoU, F1-Score, and latency comparison.
    """
    import time
    payload = request.get_json(silent=True) or request.form.to_dict()
    session_id = payload.get("session_id")

    if not session_id:
        return jsonify({"success": False, "error": "Missing 'session_id'."}), 400

    clean_session_id = secure_filename(session_id)
    session_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], clean_session_id)

    if not os.path.isdir(session_dir):
        return jsonify({"success": False, "error": f"Session '{session_id}' not found."}), 404

    path_before = os.path.join(session_dir, "before.png")
    path_aligned = os.path.join(session_dir, "aligned.png")

    if not os.path.exists(path_aligned):
        path_aligned = os.path.join(session_dir, "after.png")

    img_ref = load_image_bgr(path_before)
    img_tgt = load_image_bgr(path_aligned)

    # 1. Classical SSIM
    t0 = time.perf_counter()
    _, mask_ssim, metrics_ssim = detect_changes(img_ref, img_tgt, model_type="classical", sensitivity=0.35)
    lat_ssim = round((time.perf_counter() - t0) * 1000, 2)

    # 2. Deep Learning Siamese U-Net
    t1 = time.perf_counter()
    _, mask_ml, metrics_ml = detect_changes(img_ref, img_tgt, model_type="siamese_unet", sensitivity=0.50)
    lat_ml = round((time.perf_counter() - t1) * 1000, 2)

    # Cross-Model Agreement Metrics (IoU & F1 between Classical & ML)
    cross_metrics = compute_change_metrics(mask_ml, mask_ssim)

    return jsonify({
        "success": True,
        "session_id": session_id,
        "classical": {
            "model": "Classical SSIM + Morphological Contours",
            "latency_ms": lat_ssim,
            "change_pct": metrics_ssim.get("change_percentage", 0),
            "regions_count": metrics_ssim.get("changed_regions_count", 0),
        },
        "deep_learning": {
            "model": "Siamese U-Net (FC-Siam-diff)",
            "latency_ms": lat_ml,
            "change_pct": metrics_ml.get("change_percentage", 0),
            "regions_count": metrics_ml.get("changed_regions_count", 0),
        },
        "agreement_metrics": {
            "iou_overlap": cross_metrics["iou"],
            "f1_agreement": cross_metrics["f1_score"],
            "precision": cross_metrics["precision"],
            "recall": cross_metrics["recall"],
            "overall_accuracy": cross_metrics["overall_accuracy"],
        },
    }), 200

@main_bp.route("/uploads/<session_id>/<filename>")
def serve_upload(session_id, filename):
    """Safely serves uploaded frames from session storage."""
    session_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], secure_filename(session_id))
    return send_from_directory(session_dir, secure_filename(filename))

@main_bp.route("/samples/<filename>")
def serve_sample(filename):
    """Serves bundled sample satellite imagery for testing."""
    samples_dir = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__))), "samples")
    return send_from_directory(samples_dir, secure_filename(filename))

@main_bp.route("/api/report/<session_id>/pdf", methods=["GET"])
def download_report_pdf(session_id):
    """
    Compiles and streams a publication-grade GEOINT change intelligence PDF report.
    """
    clean_session_id = secure_filename(session_id)
    session_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], clean_session_id)

    if not os.path.isdir(session_dir):
        return jsonify({
            "success": False,
            "error": f"Session '{session_id}' not found or expired.",
        }), 404

    try:
        pdf_path = generate_pdf_report(session_dir=session_dir, session_id=clean_session_id)
        download_filename = f"ORBIT_GEOINT_Report_{clean_session_id}.pdf"
        return send_file(
            pdf_path,
            mimetype="application/pdf",
            as_attachment=True,
            download_name=download_filename,
        )
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Failed to generate PDF report: {str(e)}",
        }), 500
