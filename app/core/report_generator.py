"""
PDF Report Generator for the ORBIT Platform.

Generates professional, publication-grade Geospatial Intelligence (GEOINT)
change detection reports using ReportLab.
"""

import os
import json
import hashlib
import datetime
from typing import Dict, Any, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas
from PIL import Image as PILImage


class NumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas to dynamically compute and stamp total page count ('Page X of Y')
    along with running header and footer decoration on every page.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, page_count: int):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))

        width, height = A4
        margin = 36  # 0.5 inch

        # Running Header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(margin, height - 25, "ORBIT // SATELLITE BI-TEMPORAL CHANGE INTELLIGENCE REPORT")
            self.drawRightString(width - margin, height - 25, f"MISSION REF: {getattr(self, 'session_id', 'ORBIT-TASK')}")
            self.setStrokeColor(colors.HexColor("#CBD5E1"))
            self.setLineWidth(0.5)
            self.line(margin, height - 28, width - margin, height - 28)

        # Running Footer (all pages)
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(margin, 28, width - margin, 28)

        self.drawString(margin, 18, "ORBIT PLATFORM v1.0 • EARTH OBSERVATION AI • OFFICIAL TELEMETRY RECORD")
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(width - margin, 18, page_str)

        self.restoreState()


def get_scaled_image(img_path: str, max_width: float, max_height: float) -> Optional[RLImage]:
    """Loads an image and calculates proportional dimensions to fit max_width and max_height."""
    if not os.path.exists(img_path):
        return None
    try:
        with PILImage.open(img_path) as p_img:
            orig_w, orig_h = p_img.size
            if orig_w == 0 or orig_h == 0:
                return None
            aspect = orig_h / orig_w
            
            # Fit within bounding box
            w = max_width
            h = w * aspect
            if h > max_height:
                h = max_height
                w = h / aspect

            return RLImage(img_path, width=w, height=h)
    except Exception:
        return None


def generate_pdf_report(session_dir: str, session_id: str, results_data: Optional[Dict[str, Any]] = None) -> str:
    """
    Compiles a comprehensive GEOINT change intelligence PDF report for the given session.

    Args:
        session_dir: Absolute path to the session folder holding images and results.
        session_id: Session identifier string.
        results_data: Optional dictionary containing detection and alignment metrics.
                      If None, reads from session_dir/analysis_results.json.

    Returns:
        Absolute filepath to the generated PDF.
    """
    pdf_filename = f"ORBIT_Report_{session_id}.pdf"
    pdf_path = os.path.join(session_dir, pdf_filename)

    # 1. Load results metadata
    if not results_data:
        json_path = os.path.join(session_dir, "analysis_results.json")
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                results_data = json.load(f)
        else:
            results_data = {}

    detection = results_data.get("detection", {})
    alignment = results_data.get("alignment", {})
    pipeline = results_data.get("pipeline", "CLASSICAL_ORB_SSIM_V1")
    model_type = results_data.get("model_type", "classical")
    created_at = results_data.get("created_at", datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))

    # Severity styling
    severity = str(detection.get("severity", "MODERATE")).upper()
    if severity == "HIGH":
        severity_color = colors.HexColor("#DC2626")
        severity_bg = colors.HexColor("#FEF2F2")
    elif severity == "MODERATE":
        severity_color = colors.HexColor("#D97706")
        severity_bg = colors.HexColor("#FFFBEB")
    elif severity == "LOW":
        severity_color = colors.HexColor("#2563EB")
        severity_bg = colors.HexColor("#EFF6FF")
    else:  # STABLE
        severity_color = colors.HexColor("#059669")
        severity_bg = colors.HexColor("#ECFDF5")

    # Document setup: A4 with 36pt margins
    margin = 36
    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
    )

    # Set canvas session_id for header
    NumberedCanvas.session_id = session_id

    # Styles
    base_styles = getSampleStyleSheet()
    
    style_title = ParagraphStyle(
        "ReportTitle",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0F172A"),
    )
    
    style_subtitle = ParagraphStyle(
        "ReportSubtitle",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#64748B"),
    )

    style_section_heading = ParagraphStyle(
        "SectionHeading",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1E293B"),
        spaceBefore=8,
        spaceAfter=4,
    )

    style_body = ParagraphStyle(
        "ReportBody",
        parent=base_styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
    )

    style_body_bold = ParagraphStyle(
        "ReportBodyBold",
        parent=style_body,
        fontName="Helvetica-Bold",
    )

    style_caption = ParagraphStyle(
        "ImageCaption",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        alignment=1,  # Centered
        textColor=colors.HexColor("#475569"),
    )

    style_badge = ParagraphStyle(
        "SeverityBadge",
        parent=base_styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        alignment=1,
        textColor=severity_color,
    )

    flowables = []

    # ========================================================
    # 1. HEADER & META BANNER
    # ========================================================
    header_table_data = [
        [
            Paragraph("ORBIT // GEOINT INTELLIGENCE", style_subtitle),
            Paragraph(f"CLASSIFICATION: <b>OFFICIAL / TELEMETRY</b>", ParagraphStyle("RightSub", parent=style_subtitle, alignment=2)),
        ],
        [
            Paragraph("Bi-Temporal Satellite Change Report", style_title),
            Paragraph(f"<b>SESSION ID:</b> {session_id}<br/><b>ANALYZED:</b> {created_at}", ParagraphStyle("RightMeta", parent=style_body, alignment=2, fontSize=8, leading=11)),
        ],
    ]
    header_table = Table(header_table_data, colWidths=[340, 183])
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
    ]))
    flowables.append(header_table)
    flowables.append(Spacer(1, 8))
    flowables.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563EB"), spaceBefore=2, spaceAfter=8))

    # ========================================================
    # 2. EXECUTIVE INTELLIGENCE SUMMARY
    # ========================================================
    headline = detection.get("headline", "Satellite Observation Analysis")
    exec_summary = detection.get("executive_summary", "Surface analysis complete.")
    primary_driver = detection.get("primary_driver", "Physical Terrain Variance")

    summary_content = [
        [
            Paragraph(f"<b>SEVERITY LEVEL:</b> {severity}", style_badge),
            Paragraph(f"<b>PRIMARY DRIVER:</b> {primary_driver}", ParagraphStyle("DriverStyle", parent=style_body, fontName="Helvetica-Bold", textColor=colors.HexColor("#0F172A"))),
        ],
        [
            Paragraph(f"<b>{headline}</b>", ParagraphStyle("HL", parent=style_body, fontName="Helvetica-Bold", fontSize=10, leading=14, textColor=colors.HexColor("#0F172A"))),
            "",
        ],
        [
            Paragraph(exec_summary, style_body),
            "",
        ],
    ]
    summary_table = Table(summary_content, colWidths=[140, 383])
    summary_table.setStyle(TableStyle([
        ("SPAN", (0, 1), (1, 1)),
        ("SPAN", (0, 2), (1, 2)),
        ("BACKGROUND", (0, 0), (-1, -1), severity_bg),
        ("BOX", (0, 0), (-1, -1), 1, severity_color),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]))
    flowables.append(summary_table)
    flowables.append(Spacer(1, 10))

    # ========================================================
    # 3. KEY PERFORMANCE METRICS MATRIX
    # ========================================================
    change_pct = detection.get("change_percentage", 0.0)
    total_px_changed = detection.get("total_changed_pixels", 0)
    regions_count = detection.get("changed_regions_count", 0)
    ssim_score = detection.get("ssim_similarity_score", "N/A")
    divergence = detection.get("overall_divergence", "N/A")
    align_status = alignment.get("status", "BYPASSED")
    inliers = alignment.get("inliers_count", 0)
    sens = detection.get("sensitivity_applied", 0.35)

    metrics_data = [
        [
            Paragraph(f"<b>Total Surface Changed:</b><br/><font size=12 color='#0F172A'><b>{change_pct}%</b></font><br/><font size=7 color='#64748B'>{total_px_changed:,} altered sq. pixels</font>", style_body),
            Paragraph(f"<b>Active Change Clusters:</b><br/><font size=12 color='#0F172A'><b>{regions_count} Zones</b></font><br/><font size=7 color='#64748B'>Segmented contour clusters</font>", style_body),
            Paragraph(f"<b>Structural Divergence:</b><br/><font size=12 color='#0F172A'><b>{divergence}</b></font><br/><font size=7 color='#64748B'>SSIM Score: {ssim_score}</font>", style_body),
            Paragraph(f"<b>Spatial Alignment:</b><br/><font size=12 color='#0F172A'><b>{align_status}</b></font><br/><font size=7 color='#64748B'>{inliers} ORB homography inliers</font>", style_body),
        ]
    ]
    metrics_table = Table(metrics_data, colWidths=[130, 131, 131, 131])
    metrics_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    flowables.append(metrics_table)
    flowables.append(Spacer(1, 10))

    # ========================================================
    # 4. SATELLITE IMAGERY EVIDENCE (4-PANEL QUADRANT)
    # ========================================================
    flowables.append(Paragraph("<b>Bi-Temporal Satellite Imagery Evidence</b>", style_section_heading))

    # Locate image files
    files = os.listdir(session_dir) if os.path.exists(session_dir) else []
    before_file = next((f for f in files if f.startswith("before.")), "before.png")
    after_file = next((f for f in files if f.startswith("after.")), "after.png")
    
    path_before = os.path.join(session_dir, before_file)
    path_after = os.path.join(session_dir, after_file)
    path_overlay = os.path.join(session_dir, "diff_overlay.png")
    path_heatmap = os.path.join(session_dir, "diff_heatmap.png")

    img_box_w = 252  # Half width minus spacing
    img_box_h = 160

    img_before = get_scaled_image(path_before, img_box_w, img_box_h)
    img_after = get_scaled_image(path_after, img_box_w, img_box_h)
    img_overlay = get_scaled_image(path_overlay, img_box_w, img_box_h)
    img_heatmap = get_scaled_image(path_heatmap, img_box_w, img_box_h)

    def wrap_cell(img_obj, caption_text):
        if img_obj:
            return [img_obj, Spacer(1, 2), Paragraph(caption_text, style_caption)]
        return [Paragraph(f"[Image: {caption_text} - Not Available]", style_caption)]

    evidence_data = [
        [
            wrap_cell(img_before, "1. Reference Baseline (T-0)"),
            wrap_cell(img_after, "2. Target Observation (T-1)"),
        ],
        [
            wrap_cell(img_overlay, "3. Detected Alterations Overlay (Red)"),
            wrap_cell(img_heatmap, "4. Thermal Divergence Heatmap"),
        ]
    ]

    evidence_table = Table(evidence_data, colWidths=[258, 258])
    evidence_table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F1F5F9")),
        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#CBD5E1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
    ]))
    flowables.append(evidence_table)
    flowables.append(Spacer(1, 12))

    # ========================================================
    # 5. DETAILED CHANGE BREAKDOWN MANIFEST TABLE
    # ========================================================
    boxes = detection.get("bounding_boxes", [])
    if boxes:
        manifest_flowables = []
        manifest_flowables.append(Paragraph(f"<b>Detailed Zone Manifest ({len(boxes)} Locations Identified)</b>", style_section_heading))

        table_rows = [
            [
                Paragraph("<b>Zone ID</b>", style_body_bold),
                Paragraph("<b>Footprint (px)</b>", style_body_bold),
                Paragraph("<b>Share (%)</b>", style_body_bold),
                Paragraph("<b>Classification Tag</b>", style_body_bold),
                Paragraph("<b>Confidence</b>", style_body_bold),
                Paragraph("<b>Bounding Box [X, Y, W, H]</b>", style_body_bold),
            ]
        ]

        # Display top 10 zones in the report
        for b in boxes[:10]:
            label = b.get("label", f"Zone #{b.get('id', 1)}")
            area = f"{b.get('area_px', 0):,}"
            share = f"{b.get('share_pct', 0)}%"
            tag = b.get("tag", "Structural Alteration")
            conf = b.get("confidence", "High")
            coords = f"[{b.get('x')}, {b.get('y')}, {b.get('width')}, {b.get('height')}]"

            table_rows.append([
                Paragraph(label, style_body),
                Paragraph(area, style_body),
                Paragraph(share, style_body),
                Paragraph(tag, style_body),
                Paragraph(conf, style_body),
                Paragraph(coords, style_body),
            ])

        if len(boxes) > 10:
            table_rows.append([
                Paragraph(f"<i>... and {len(boxes) - 10} additional smaller zones omitted for brevity.</i>", style_body),
                "", "", "", "", ""
            ])

        zone_table = Table(table_rows, colWidths=[75, 75, 60, 125, 75, 113])
        zone_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FAFC")]),
            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#CBD5E1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ]))
        if len(boxes) > 10:
            zone_table.setStyle(TableStyle([
                ("SPAN", (0, -1), (-1, -1)),
            ]))

        manifest_flowables.append(zone_table)
        flowables.append(KeepTogether(manifest_flowables))
        flowables.append(Spacer(1, 10))

    # ========================================================
    # 6. METHODOLOGY & AUDIT TRAIL
    # ========================================================
    method_text = (
        f"<b>Methodology & Calibration:</b> Processed via ORBIT {pipeline} architecture with sensitivity "
        f"threshold {sens:.2f} and morphological contour filtering. Images spatially calibrated via ORB feature "
        f"homography estimation. Sensor integrity verified prior to tensor ingestion."
    )
    flowables.append(Paragraph(method_text, ParagraphStyle("MethodStyle", parent=style_body, fontSize=7.5, leading=10, textColor=colors.HexColor("#64748B"))))

    # Build document
    doc.build(flowables, canvasmaker=NumberedCanvas)
    return pdf_path
