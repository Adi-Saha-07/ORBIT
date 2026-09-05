/**
 * ORBIT // Telemetry, Ingestion & Multi-Layer Visualizer Controller
 * Optimized for Instant Human Understanding & Professional GEOINT Intelligence
 */

document.addEventListener("DOMContentLoaded", () => {
    // ========================================================
    // DOM REFERENCES
    // ========================================================
    // Global Header & Status
    const sysStatus = document.getElementById("sys-status");
    const pipelineStatus = document.getElementById("pipeline-status");
    const sessionBadge = document.getElementById("session-badge");
    const consoleLogs = document.getElementById("console-logs");

    // Sections
    const sectionIngestion = document.getElementById("section-ingestion");
    const sectionInspection = document.getElementById("section-inspection");

    // Ingestion Form & Controls
    const form = document.getElementById("upload-form");
    const btnSubmit = document.getElementById("btn-submit-upload");
    const btnUploadText = document.getElementById("btn-upload-text");
    const btnClearAll = document.getElementById("btn-clear-all");
    const btnLoadSample = document.getElementById("btn-load-sample");

    // Sensor Alpha (Before / T-0)
    const dropBefore = document.getElementById("dropzone-before");
    const inputBefore = document.getElementById("file-before");
    const contentBefore = document.getElementById("content-before");
    const previewContBefore = document.getElementById("preview-container-before");
    const previewImgBefore = document.getElementById("preview-img-before");
    const metaBefore = document.getElementById("meta-before");
    const nameBefore = document.getElementById("file-name-before");
    const sizeBefore = document.getElementById("file-size-before");
    const btnRemoveBefore = document.getElementById("btn-remove-before");

    // Sensor Beta (After / T-1)
    const dropAfter = document.getElementById("dropzone-after");
    const inputAfter = document.getElementById("file-after");
    const contentAfter = document.getElementById("content-after");
    const previewContAfter = document.getElementById("preview-container-after");
    const previewImgAfter = document.getElementById("preview-img-after");
    const metaAfter = document.getElementById("meta-after");
    const nameAfter = document.getElementById("file-name-after");
    const sizeAfter = document.getElementById("file-size-after");
    const btnRemoveAfter = document.getElementById("btn-remove-after");

    // Executive Summary Elements
    const insightSeverityPill = document.getElementById("insight-severity-pill");
    const insightDriverPill = document.getElementById("insight-driver-pill");
    const insightHeadline = document.getElementById("insight-headline");
    const insightSummary = document.getElementById("insight-summary");

    // Stage 3 Inspection Elements
    const btnBackToIngest = document.getElementById("btn-back-to-ingest");
    const sensitivitySlider = document.getElementById("sensitivity-slider");
    const sensitivityVal = document.getElementById("sensitivity-val");
    const btnReanalyze = document.getElementById("btn-reanalyze");
    const viewTabs = document.querySelectorAll(".tab-pill, .tab-btn");

    // KPI Metrics
    const kpiChangePct = document.getElementById("kpi-change-pct");
    const kpiChangePx = document.getElementById("kpi-change-px");
    const kpiRegionsCount = document.getElementById("kpi-regions-count");
    const kpiSsimScore = document.getElementById("kpi-ssim-score");
    const kpiSsimDivergence = document.getElementById("kpi-ssim-divergence");
    const kpiAlignmentStatus = document.getElementById("kpi-alignment-status");
    const kpiInliersRatio = document.getElementById("kpi-inliers-ratio");

    // Viewports
    const viewCurtain = document.getElementById("view-curtain");
    const viewLayer = document.getElementById("view-layer");
    const viewSideBySide = document.getElementById("view-sidebyside");

    // Curtain Elements
    const curtainWrapper = document.getElementById("curtain-wrapper");
    const curtainClip = document.getElementById("curtain-clip");
    const curtainHandle = document.getElementById("curtain-handle");
    const curtainImgBefore = document.getElementById("curtain-img-before");
    const curtainImgAfter = document.getElementById("curtain-img-after");

    // Layer Elements
    const layerActiveImg = document.getElementById("layer-active-img");
    const layerHeatmapImg = document.getElementById("layer-heatmap-img");
    const layerBtns = document.querySelectorAll(".btn-layer");
    const opacitySlider = document.getElementById("opacity-slider");
    const opacityVal = document.getElementById("opacity-val");
    const opacityGroup = document.getElementById("opacity-control-group");

    // 3-Way Side-by-Side Elements
    const dualImgBefore = document.getElementById("dual-img-before");
    const dualImgTarget = document.getElementById("dual-img-target");
    const dualImgAfter = document.getElementById("dual-img-after");

    // Manifest Elements
    const manifestCount = document.getElementById("manifest-count");
    const manifestList = document.getElementById("manifest-list");

    // ========================================================
    // STATE MANAGEMENT
    // ========================================================
    let currentSessionId = null;
    let currentAnalysisData = null;
    let isDraggingCurtain = false;
    let activeModelType = "classical"; // Default to instant, robust Classical CV for cloud compatibility

    const fileState = {
        before: null,
        after: null,
    };

    // ========================================================
    // LOGGING UTILITY
    // ========================================================
    function log(message, type = "info") {
        console.log(`[ORBIT] [${type.toUpperCase()}] ${message}`);
        if (consoleLogs) {
            const time = new Date().toTimeString().split(" ")[0];
            const entry = document.createElement("div");
            entry.className = `log-entry ${type}`;
            entry.innerHTML = `<span class="log-time">[${time}]</span> ${message}`;
            consoleLogs.appendChild(entry);
            consoleLogs.scrollTop = consoleLogs.scrollHeight;
        }
    }

    function formatBytes(bytes) {
        if (bytes === 0) return "0.00 KB";
        const k = 1024;
        const dm = 2;
        const sizes = ["Bytes", "KB", "MB", "GB"];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
    }

    // Ping Health Check
    async function checkEngineHealth() {
        try {
            const res = await fetch("/api/health");
            if (res.ok) {
                const data = await res.json();
                sysStatus.textContent = "ONLINE";
                log(`Telemetry link verified with ${data.system} [v${data.version}]`, "success");
            }
        } catch (e) {
            sysStatus.textContent = "OFFLINE";
            log("Telemetry engine offline.", "error");
        }
    }

    // ========================================================
    // DRAG-AND-DROP INGESTION
    // ========================================================
    function setupDropzone(dropzone, input, type) {
        ["dragenter", "dragover"].forEach(evt => {
            dropzone.addEventListener(evt, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add("drag-over");
            });
        });

        ["dragleave", "drop"].forEach(evt => {
            dropzone.addEventListener(evt, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove("drag-over");
            });
        });

        dropzone.addEventListener("drop", (e) => {
            const files = e.dataTransfer.files;
            if (files && files.length > 0) {
                processFile(files[0], type);
            }
        });

        input.addEventListener("change", (e) => {
            if (e.target.files && e.target.files.length > 0) {
                processFile(e.target.files[0], type);
            }
        });
    }

    function processFile(file, type) {
        const isBefore = type === "before";
        const content = isBefore ? contentBefore : contentAfter;
        const previewCont = isBefore ? previewContBefore : previewContAfter;
        const previewImg = isBefore ? previewImgBefore : previewImgAfter;
        const meta = isBefore ? metaBefore : metaAfter;
        const nameLabel = isBefore ? nameBefore : nameAfter;
        const sizeLabel = isBefore ? sizeBefore : sizeAfter;

        const ext = file.name.split(".").pop().toLowerCase();
        if (!["png", "jpg", "jpeg", "tif", "tiff"].includes(ext)) {
            alert(`'${file.name}' is not supported. Please upload PNG, JPG, or TIFF satellite images.`);
            return;
        }

        const reader = new FileReader();
        reader.onload = (e) => {
            const tempImg = new Image();
            tempImg.onload = () => {
                const w = tempImg.naturalWidth;
                const h = tempImg.naturalHeight;

                previewImg.src = e.target.result;
                meta.textContent = `${w} × ${h} px [${ext.toUpperCase()}]`;
                nameLabel.textContent = file.name;
                sizeLabel.textContent = formatBytes(file.size);

                content.style.display = "none";
                previewCont.classList.remove("hidden");

                fileState[type] = file;
                updateSubmitState();
            };
            tempImg.src = e.target.result;
        };
        reader.readAsDataURL(file);
    }

    function clearSensor(type) {
        const isBefore = type === "before";
        const content = isBefore ? contentBefore : contentAfter;
        const previewCont = isBefore ? previewContBefore : previewContAfter;
        const previewImg = isBefore ? previewImgBefore : previewImgAfter;
        const nameLabel = isBefore ? nameBefore : nameAfter;
        const sizeLabel = isBefore ? sizeBefore : sizeAfter;
        const input = isBefore ? inputBefore : inputAfter;

        fileState[type] = null;
        input.value = "";
        previewImg.src = "";
        previewCont.classList.add("hidden");
        content.style.display = "flex";
        nameLabel.textContent = "No file selected";
        sizeLabel.textContent = "0.00 KB";

        updateSubmitState();
    }

    btnRemoveBefore.addEventListener("click", (e) => {
        e.stopPropagation();
        clearSensor("before");
    });

    btnRemoveAfter.addEventListener("click", (e) => {
        e.stopPropagation();
        clearSensor("after");
    });

    btnClearAll.addEventListener("click", () => {
        clearSensor("before");
        clearSensor("after");
        currentSessionId = null;
        currentAnalysisData = null;
        sessionBadge.textContent = "SESSION: READY";
        updateSubmitState();
    });

    function updateSubmitState() {
        const ready = fileState.before !== null && fileState.after !== null;
        btnSubmit.disabled = !ready;
    }

    // ========================================================
    // 1-CLICK SAMPLE SATELLITE PAIR LOADER
    // ========================================================
    btnLoadSample.addEventListener("click", async () => {
        btnLoadSample.disabled = true;
        btnLoadSample.innerHTML = `<span>Loading Scenario...</span>`;

        try {
            const [res0, res1] = await Promise.all([
                fetch("/samples/satellite_t0_baseline.png"),
                fetch("/samples/satellite_t1_target.png"),
            ]);

            if (!res0.ok || !res1.ok) throw new Error("Could not fetch sample satellite imagery.");

            const blob0 = await res0.blob();
            const blob1 = await res1.blob();

            const file0 = new File([blob0], "satellite_t0_baseline.png", { type: "image/png" });
            const file1 = new File([blob1], "satellite_t1_target.png", { type: "image/png" });

            processFile(file0, "before");
            processFile(file1, "after");
            log("Sample scenario loaded. Click 'Analyze Satellite Changes' to run detection.", "success");
        } catch (err) {
            alert(`Sample load failed: ${err.message}`);
        } finally {
            btnLoadSample.disabled = false;
            btnLoadSample.innerHTML = `
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="btn-icon">
                    <path d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"/>
                </svg>
                <span>Load Demo Satellite Pair</span>
            `;
        }
    });

    // Ensure clicking submit button triggers upload on all mobile browsers
    if (btnSubmit) {
        btnSubmit.addEventListener("click", (e) => {
            if (form) {
                if (form.requestSubmit) {
                    form.requestSubmit();
                } else {
                    form.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
                }
            }
        });
    }

    // ========================================================
    // 1-CLICK COMBINED SUBMISSION & INSTANT ANALYSIS
    // ========================================================
    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        if (!fileState.before || !fileState.after) {
            alert("Please provide both Before (T-0) and After (T-1) satellite images.");
            return;
        }

        btnSubmit.disabled = true;
        if (btnUploadText) btnUploadText.textContent = "Uploading & Aligning...";

        const formData = new FormData();
        formData.append("image_before", fileState.before);
        formData.append("image_after", fileState.after);

        try {
            const response = await fetch("/api/upload", {
                method: "POST",
                body: formData,
            });

            let data;
            const contentType = response.headers.get("content-type");
            if (contentType && contentType.includes("application/json")) {
                data = await response.json();
            } else {
                throw new Error(`Server returned HTTP ${response.status} (${response.statusText || 'Error'}). The server may still be initializing.`);
            }

            if (response.ok && data.success) {
                currentSessionId = data.session_id;
                sessionBadge.textContent = `SESSION: ${data.session_id.substring(0, 8).toUpperCase()}`;

                if (btnUploadText) btnUploadText.textContent = "Running Analysis...";
                await executeAnalysis(0.35, activeModelType);
            } else {
                alert(data.error || "Upload failed. Please check image formats.");
            }
        } catch (err) {
            alert("Network error: " + err.message);
        } finally {
            btnSubmit.disabled = false;
            if (btnUploadText) btnUploadText.textContent = "Analyze Satellite Changes";
            updateSubmitState();
        }
    });

    // ========================================================
    // ANALYSIS PIPELINE EXECUTION
    // ========================================================
    async function executeAnalysis(sensitivity = 0.35, modelType = activeModelType) {
        if (!currentSessionId) {
            alert("No active satellite session. Please upload images first.");
            return;
        }

        btnReanalyze.disabled = true;
        pipelineStatus.textContent = "ANALYZING...";

        try {
            const res = await fetch("/api/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    session_id: currentSessionId,
                    sensitivity: sensitivity,
                    min_contour_area: 120,
                    enable_alignment: true,
                    model_type: modelType,
                }),
            });

            let data;
            const contentType = res.headers.get("content-type");
            if (contentType && contentType.includes("application/json")) {
                data = await res.json();
            } else {
                throw new Error(`Server returned HTTP ${res.status} (${res.statusText || 'Error'}). If using free cloud hosting, the instance may be constrained.`);
            }

            if (res.ok && data.success) {
                currentAnalysisData = data;
                if (data.fallback_applied) {
                    activeModelType = "classical";
                    modelTabs.forEach((t) => {
                        t.classList.toggle("active", t.getAttribute("data-model") === "classical");
                    });
                }
                pipelineStatus.textContent = data.model_type === "siamese_unet" ? "SIAMESE AI" : "CLASSICAL CV";

                // Populate and activate results view
                populateInspectionDashboard(data);
                sectionIngestion.classList.add("hidden");
                sectionInspection.classList.remove("hidden");
                window.scrollTo({ top: 0, behavior: "smooth" });
            } else {
                alert("Analysis failed: " + (data.error || "Unknown server error"));
            }
        } catch (err) {
            alert("Analysis execution failed: " + err.message);
        } finally {
            btnReanalyze.disabled = false;
        }
    }

    // Model Architecture Switcher
    const modelTabs = document.querySelectorAll(".btn-model-tab");
    modelTabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            modelTabs.forEach((t) => t.classList.remove("active"));
            tab.classList.add("active");
            activeModelType = tab.getAttribute("data-model");
            executeAnalysis(parseFloat(sensitivitySlider.value), activeModelType);
        });
    });

    // Sensitivity Cutoff Slider
    sensitivitySlider.addEventListener("input", (e) => {
        sensitivityVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    btnReanalyze.addEventListener("click", () => {
        const sens = parseFloat(sensitivitySlider.value);
        executeAnalysis(sens, activeModelType);
    });

    // Benchmark Dialog
    const btnRunBenchmark = document.getElementById("btn-run-benchmark");
    const benchmarkPanel = document.getElementById("benchmark-panel");
    const btnCloseBenchmark = document.getElementById("btn-close-benchmark");
    const benchmarkBody = document.getElementById("benchmark-body");

    if (btnRunBenchmark) {
        btnRunBenchmark.addEventListener("click", async () => {
            if (!currentSessionId) return;

            benchmarkPanel.classList.remove("hidden");
            benchmarkBody.innerHTML = `<div style="padding: 1.5rem; text-align: center; color: var(--text-secondary);">Comparing Classical CV vs Siamese U-Net Deep Learning...</div>`;

            try {
                const res = await fetch("/api/benchmark", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ session_id: currentSessionId }),
                });

                const bData = await res.json();
                if (res.ok && bData.success) {
                    const c = bData.classical;
                    const m = bData.siamese_unet;
                    const a = bData.agreement;

                    benchmarkBody.innerHTML = `
                        <table class="benchmark-table">
                            <thead>
                                <tr>
                                    <th>Metric</th>
                                    <th>Classical SSIM Baseline</th>
                                    <th>Siamese U-Net (Deep Learning)</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td><strong>Inference Time</strong></td>
                                    <td>${c.latency_ms.toFixed(1)} ms</td>
                                    <td>${m.latency_ms.toFixed(1)} ms</td>
                                </tr>
                                <tr>
                                    <td><strong>Detected Change Area</strong></td>
                                    <td>${c.change_percentage}%</td>
                                    <td>${m.change_percentage}%</td>
                                </tr>
                                <tr>
                                    <td><strong>Identified Zones</strong></td>
                                    <td>${c.changed_regions_count} Zones</td>
                                    <td>${m.changed_regions_count} Zones</td>
                                </tr>
                                <tr>
                                    <td><strong>Cross-Model Agreement (IoU)</strong></td>
                                    <td colspan="2" class="benchmark-badge-score" style="text-align: center;">${(a.iou * 100).toFixed(1)}% Overlap (F1: ${(a.f1_score * 100).toFixed(1)}%)</td>
                                </tr>
                            </tbody>
                        </table>
                    `;
                }
            } catch (err) {
                benchmarkBody.innerHTML = `<div style="color: var(--rose); padding: 1rem;">Benchmark error: ${err.message}</div>`;
            }
        });
    }

    if (btnCloseBenchmark) {
        btnCloseBenchmark.addEventListener("click", () => {
            benchmarkPanel.classList.add("hidden");
        });
    }

    // ========================================================
    // GEOINT PDF REPORT DOWNLOAD HANDLER
    // ========================================================
    const btnDownloadPdf = document.getElementById("btn-download-pdf");

    async function triggerPdfDownload(sourceBtn) {
        if (!currentSessionId) {
            alert("No active satellite session to generate report. Please run analysis first.");
            return;
        }

        const originalHtml = sourceBtn ? sourceBtn.innerHTML : null;
        if (sourceBtn) {
            sourceBtn.disabled = true;
            sourceBtn.innerHTML = `
                <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="animation: spin 1s linear infinite;">
                    <circle cx="12" cy="12" r="10" stroke-opacity="0.25"></circle>
                    <path d="M12 2a10 10 0 0 1 10 10" stroke-linecap="round"></path>
                </svg>
                <span>Generating PDF...</span>
            `;
        }

        try {
            log(`Compiling publication-grade GEOINT PDF report for session ${currentSessionId}...`, "info");
            const response = await fetch(`/api/report/${currentSessionId}/pdf`);
            if (!response.ok) {
                const errData = await response.json().catch(() => ({}));
                throw new Error(errData.error || `Server returned HTTP ${response.status}`);
            }

            const blob = await response.blob();
            const blobUrl = window.URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.style.display = "none";
            a.href = blobUrl;
            a.download = `ORBIT_GEOINT_Report_${currentSessionId}.pdf`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(blobUrl);
            a.remove();

            log("GEOINT intelligence PDF report successfully compiled and downloaded.", "success");
        } catch (err) {
            alert(`Failed to download PDF report: ${err.message}`);
            log(`PDF generation error: ${err.message}`, "error");
        } finally {
            if (sourceBtn && originalHtml) {
                sourceBtn.disabled = false;
                sourceBtn.innerHTML = originalHtml;
            }
        }
    }

    if (btnDownloadPdf) {
        btnDownloadPdf.addEventListener("click", () => triggerPdfDownload(btnDownloadPdf));
    }

    // Back to Ingestion button
    btnBackToIngest.addEventListener("click", () => {
        sectionInspection.classList.add("hidden");
        sectionIngestion.classList.remove("hidden");
        pipelineStatus.textContent = "READY";
        window.scrollTo({ top: 0, behavior: "smooth" });
    });

    // ========================================================
    // POPULATE PROFESSIONAL INSPECTION DASHBOARD
    // ========================================================
    function populateInspectionDashboard(data) {
        const det = data.detection;
        const align = data.alignment;
        const art = data.artifacts;
        const ts = Date.now();

        // 1. Populate Executive Intelligence Summary Card
        if (insightSeverityPill) insightSeverityPill.textContent = `${det.severity || "HIGH"} SEVERITY`;
        if (insightDriverPill) insightDriverPill.textContent = det.primary_driver || "New Structural Development";
        if (insightHeadline) insightHeadline.textContent = det.headline || "Major Landscape Modification Detected";
        if (insightSummary) insightSummary.textContent = det.executive_summary || "Comparing satellite captures revealed significant new physical developments across the monitored sectors.";

        // 2. Populate Intuitive KPI Cards
        kpiChangePct.textContent = `${det.change_percentage}%`;
        kpiChangePx.textContent = `${det.total_changed_pixels.toLocaleString()} sq. px altered`;
        kpiRegionsCount.textContent = `${det.changed_regions_count} Zones`;
        kpiSsimScore.textContent = det.severity || "HIGH";
        kpiSsimDivergence.textContent = det.primary_driver ? det.primary_driver.substring(0, 30) : "Significant Activity";
        kpiAlignmentStatus.textContent = data.model_type === "siamese_unet" ? "SIAMESE AI" : "CLASSICAL CV";
        kpiInliersRatio.textContent = align.status === "LOCKED" ? "Auto-Aligned & Calibrated" : "Direct Analysis";

        // 3. 3-Way Side-by-Side Images (Before | After | What Changed)
        if (dualImgBefore) dualImgBefore.src = `${art.reference_url}?t=${ts}`;
        if (dualImgTarget) dualImgTarget.src = `${art.target_url}?t=${ts}`;
        if (dualImgAfter) dualImgAfter.src = `${art.overlay_url}?t=${ts}`;

        // 4. Populate Curtain Images
        curtainImgBefore.src = `${art.reference_url}?t=${ts}`;
        curtainImgAfter.src = `${art.overlay_url}?t=${ts}`;
        setCurtainPercentage(50);

        // 5. Populate Layer Inspector Images
        layerActiveImg.src = `${art.overlay_url}?t=${ts}`;
        layerHeatmapImg.src = `${art.heatmap_url}?t=${ts}`;
        layerHeatmapImg.classList.add("hidden");

        // 6. Populate Manifest Cards with Human-Interpretable Zone Info
        manifestCount.textContent = `${det.changed_regions_count} Location${det.changed_regions_count !== 1 ? 's' : ''}`;
        manifestList.innerHTML = "";

        if (det.bounding_boxes && det.bounding_boxes.length > 0) {
            det.bounding_boxes.forEach((box) => {
                const card = document.createElement("div");
                card.className = "zone-card";
                card.innerHTML = `
                    <div class="zone-top-row">
                        <span class="zone-title">${box.label || `Zone #${box.id}`}</span>
                        <span class="zone-share-badge">${box.share_pct || 0}% of changes</span>
                    </div>
                    <div class="zone-tag-row">
                        <span class="zone-type-badge">${box.tag || "Structural Development"}</span>
                        <span class="zone-confidence-badge">${box.confidence || "High Confidence"}</span>
                    </div>
                    <div class="zone-area-text">
                        Footprint: <strong>${box.area_px.toLocaleString()} sq. px</strong> &bull; Bounding Box: [${box.width} &times; ${box.height} px]
                    </div>
                `;
                manifestList.appendChild(card);
            });
        } else {
            manifestList.innerHTML = `<div style="grid-column: 1 / -1; padding: 1.5rem; text-align: center; color: var(--text-muted);">No structural divergence detected. The landscape is completely stable.</div>`;
        }
    }

    // ========================================================
    // VIEW MODE TABS
    // ========================================================
    viewTabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            viewTabs.forEach((t) => t.classList.remove("active"));
            tab.classList.add("active");

            const viewMode = tab.getAttribute("data-view");
            viewCurtain.classList.add("hidden");
            viewLayer.classList.add("hidden");
            viewSideBySide.classList.add("hidden");

            if (viewMode === "curtain") {
                viewCurtain.classList.remove("hidden");
            } else if (viewMode === "layer") {
                viewLayer.classList.remove("hidden");
            } else if (viewMode === "sidebyside") {
                viewSideBySide.classList.remove("hidden");
            }
        });
    });

    // ========================================================
    // ZERO-DISTORTION CURTAIN REVEAL SCANNER
    // ========================================================
    const curtainRangeSlider = document.getElementById("curtain-range-slider");
    const curtainPosVal = document.getElementById("curtain-pos-val");
    const btnCurtainModes = document.querySelectorAll(".btn-curtain-mode");
    const curtainBadgeRight = document.getElementById("curtain-badge-right");

    function updateCurtainPosition(clientX) {
        const rect = curtainWrapper.getBoundingClientRect();
        let x = clientX - rect.left;
        x = Math.max(0, Math.min(x, rect.width));

        const pct = (x / rect.width) * 100;
        setCurtainPercentage(pct);
    }

    function setCurtainPercentage(pct) {
        pct = Math.max(0, Math.min(100, pct));
        curtainWrapper.style.setProperty("--clip-pos", `${pct}%`);
        if (curtainRangeSlider) curtainRangeSlider.value = Math.round(pct);
        if (curtainPosVal) curtainPosVal.textContent = `${Math.round(pct)}%`;
    }

    if (curtainRangeSlider) {
        curtainRangeSlider.addEventListener("input", (e) => {
            setCurtainPercentage(parseFloat(e.target.value));
        });
    }

    // Toggle reveal mode between Red Changes Diff and Recent Target Frame
    btnCurtainModes.forEach((btn) => {
        btn.addEventListener("click", () => {
            if (!currentAnalysisData) return;
            btnCurtainModes.forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");

            const mode = btn.getAttribute("data-reveal");
            const art = currentAnalysisData.artifacts;
            const ts = Date.now();

            if (mode === "overlay") {
                curtainImgAfter.src = `${art.overlay_url}?t=${ts}`;
                if (curtainBadgeRight) curtainBadgeRight.textContent = "Detected Changes (In Red) ▶";
            } else if (mode === "target") {
                curtainImgAfter.src = `${art.target_url}?t=${ts}`;
                if (curtainBadgeRight) curtainBadgeRight.textContent = "Recent Satellite View (T-1) ▶";
            }
        });
    });

    curtainHandle.addEventListener("mousedown", (e) => {
        isDraggingCurtain = true;
        e.preventDefault();
    });

    curtainWrapper.addEventListener("mousedown", (e) => {
        isDraggingCurtain = true;
        updateCurtainPosition(e.clientX);
    });

    window.addEventListener("mousemove", (e) => {
        if (!isDraggingCurtain) return;
        updateCurtainPosition(e.clientX);
    });

    window.addEventListener("mouseup", () => {
        isDraggingCurtain = false;
    });

    curtainHandle.addEventListener("touchstart", (e) => {
        isDraggingCurtain = true;
        if (e.touches && e.touches[0]) {
            updateCurtainPosition(e.touches[0].clientX);
        }
    }, { passive: true });

    curtainWrapper.addEventListener("touchstart", (e) => {
        if (!e.touches || !e.touches[0]) return;
        isDraggingCurtain = true;
        updateCurtainPosition(e.touches[0].clientX);
    }, { passive: true });

    window.addEventListener("touchmove", (e) => {
        if (!isDraggingCurtain || !e.touches || !e.touches[0]) return;
        updateCurtainPosition(e.touches[0].clientX);
    }, { passive: true });

    window.addEventListener("touchend", () => {
        isDraggingCurtain = false;
    });

    window.addEventListener("touchcancel", () => {
        isDraggingCurtain = false;
    });


    // ========================================================
    // LAYER INSPECTOR TOGGLES
    // ========================================================
    layerBtns.forEach((btn) => {
        btn.addEventListener("click", () => {
            if (!currentAnalysisData) return;
            layerBtns.forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");

            const layer = btn.getAttribute("data-layer");
            const art = currentAnalysisData.artifacts;
            const ts = Date.now();

            if (layer === "overlay") {
                layerActiveImg.src = `${art.overlay_url}?t=${ts}`;
                layerHeatmapImg.classList.add("hidden");
                if (opacityGroup) opacityGroup.style.display = "none";
            } else if (layer === "heatmap") {
                layerActiveImg.src = `${art.heatmap_url}?t=${ts}`;
                layerHeatmapImg.classList.add("hidden");
                if (opacityGroup) opacityGroup.style.display = "none";
            } else if (layer === "target") {
                layerActiveImg.src = `${art.target_url}?t=${ts}`;
                layerHeatmapImg.classList.add("hidden");
                if (opacityGroup) opacityGroup.style.display = "none";
            } else if (layer === "reference") {
                layerActiveImg.src = `${art.reference_url}?t=${ts}`;
                layerHeatmapImg.classList.add("hidden");
                if (opacityGroup) opacityGroup.style.display = "none";
            }
        });
    });

    if (opacitySlider) {
        opacitySlider.addEventListener("input", (e) => {
            const val = e.target.value;
            if (opacityVal) opacityVal.textContent = `${val}%`;
            layerHeatmapImg.style.opacity = val / 100;
        });
    }

    // ========================================================
    // THEME MANAGEMENT (LIGHT / DARK - LIGHT DEFAULT)
    // ========================================================
    const themeToggleBtn = document.getElementById("theme-toggle-btn");
    const themeModeText = document.querySelector(".theme-mode-text");

    function applyTheme(theme) {
        document.documentElement.setAttribute("data-theme", theme);
        localStorage.setItem("orbit-theme", theme);
        if (themeModeText) {
            themeModeText.textContent = theme === "light" ? "Light Mode" : "Dark Mode";
        }
    }

    const savedTheme = localStorage.getItem("orbit-theme") || "light";
    applyTheme(savedTheme);

    if (themeToggleBtn) {
        themeToggleBtn.addEventListener("click", () => {
            const current = document.documentElement.getAttribute("data-theme") || "light";
            const next = current === "light" ? "dark" : "light";
            applyTheme(next);
        });
    }

    // Initial Setup
    setupDropzone(dropBefore, inputBefore, "before");
    setupDropzone(dropAfter, inputAfter, "after");
    checkEngineHealth();
});
