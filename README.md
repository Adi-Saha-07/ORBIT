# ORBIT // Enterprise Satellite Change Detection & GEOINT Intelligence

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000.svg)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**ORBIT** is an end-to-end geospatial intelligence (GEOINT) web application designed for bi-temporal satellite image change detection. Users can upload or select two satellite captures of the same geographical area taken at different timestamps ($T_0$ Baseline and $T_1$ Target). 

The platform automatically registers and aligns the imagery, runs deep learning and computer vision change detection pipelines, and delivers actionable, human-interpretable intelligence with zero-distortion interactive visualizers.

---

## 🌟 Key Features

### 1. Dual Inference Engines
* **PyTorch Siamese U-Net (`FC-Siam-diff`)**: Fully convolutional deep neural network featuring weight-shared twin encoders with skip connections that isolates true physical modifications while suppressing seasonal variations, illumination drift, and atmospheric noise.
* **Classical Computer Vision Pipeline**: Feature-based ORB alignment with RANSAC homography estimation, Structural Similarity Index Measure (SSIM), Otsu adaptive thresholding, and morphological filtering.
* **Side-by-Side Performance Benchmark**: Live head-to-head comparison modal measuring latency (ms), detected change area (%), identified zones count, and cross-model agreement (IoU and F1-Score).

### 2. Human-Centric Executive Intelligence
* **Executive Summary Card**: Translates complex pixel arrays into plain-English intelligence summaries, specifying primary drivers (e.g., *New Structural Development*), severity rating (*HIGH*, *MODERATE*, *LOW*, *STABLE*), and affected areas.
* **Change Manifest & Zone Cards**: Identifies specific geographic zones, pixel footprints, bounding boxes, and relative shares of change.

### 3. Interactive Multi-Mode Visualizers
* **Zero-Distortion Curtain Reveal Scanner**: Smooth horizontal slider using CSS `clip-path` that sweeps across satellite images without image compression, squishing, or distortion. Toggle between *Changes Diff (In Red)* and *Recent Satellite View (T-1)*.
* **3-Way Side-by-Side View**: Clear sequential layout displaying *1. Baseline (T-0)*, *2. Recent View (T-1)*, and *3. What Changed (Red Highlights)*.
* **Multi-Layer Inspector**: Heatmap intensity layer with dynamic opacity control.

### 4. Enterprise Design & Mobile Responsiveness
* **Dual Theme Engine**: Enterprise Light Mode (default) and High-Contrast Tactical Dark Mode with smooth transition tokens.
* **Full Mobile Responsiveness**: Touch-optimized interface supporting drag-to-reveal sliders, touch targets, and stacked cards on smartphones and tablets.

---

## 🏗️ System Architecture

```
                               ┌─────────────────────────────┐
                               │  Sensor Alpha (Baseline T0) │
                               │  Sensor Beta  (Target   T1) │
                               └──────────────┬──────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │  Preprocessing & Validation     │
                             │  • Format verification (PNG/JPG)│
                             │  • Dimension alignment (H x W)  │
                             └────────────────┬────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │  ORB + RANSAC Feature Aligner   │
                             │  • Sub-pixel homography matrix  │
                             │  • Geometric correction         │
                             └────────────────┬────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
      ┌─────────────────────────────┐                   ┌─────────────────────────────┐
      │   PyTorch Siamese U-Net     │                   │     Classical CV (SSIM)     │
      │   • Twin Encoder Extractors │                   │   • Luminance / Contrast    │
      │   • Absolute Difference     │                   │   • Otsu Thresholding       │
      │   • Skip Concatenations     │                   │   • Morphological Filtering │
      └──────────────┬──────────────┘                   └──────────────┬──────────────┘
                     │                                                 │
                     └────────────────────────┬────────────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │  Analytics & Intelligence       │
                             │  • Bounding box extraction      │
                             │  • Severity & driver inference  │
                             │  • Overlay & heatmap rendering  │
                             └────────────────┬────────────────┘
                                              │
                                              ▼
                             ┌─────────────────────────────────┐
                             │  Interactive Web UI             │
                             │  • Curtain Reveal Scanner       │
                             │  • 3-Way Side-by-Side           │
                             │  • Executive Summary & KPIs     │
                             └─────────────────────────────────┘
```

---

## 🚀 Getting Started

### Prerequisites
* Python 3.10, 3.11, or 3.12
* Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Adi-Saha-07/ORBIT.git
   cd ORBIT
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # On Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   # On Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the web application:**
   ```bash
   python run.py
   ```
   Open your browser and navigate to: **`http://127.0.0.1:5000/`**

---

## 🧪 Running Automated Tests

ORBIT includes an automated test suite covering OpenCV geometric alignment, SSIM change detection, Siamese U-Net forward pass, custom BCE+Dice loss functions, API endpoints, and error handling:

```bash
pytest -v
```

All 18 unit and integration tests pass out-of-the-box.

---

## 📂 Project Structure

```
ORBIT/
├── app/
│   ├── core/
│   │   ├── aligner.py        # ORB feature detection & RANSAC homography alignment
│   │   ├── detector.py       # SSIM diff engine, contour analysis, executive summaries
│   │   ├── metrics.py        # SSIM, IoU, and F1 calculation utilities
│   │   ├── ml_inference.py   # PyTorch Siamese U-Net inference wrapper
│   │   ├── preprocessor.py   # Dimension normalization and channel conversion
│   │   ├── validator.py      # Image integrity and dimension validation
│   │   └── visualizer.py     # Alpha overlays, heatmap colormaps, side-by-side exports
│   ├── static/
│   │   ├── css/style.css     # Enterprise Light/Dark theme & mobile responsive design
│   │   └── js/upload.js      # Zero-distortion slider, touch controller, API client
│   ├── templates/
│   │   └── index.html        # Main platform interface and visualizer stages
│   ├── routes.py             # Flask API endpoints (/api/upload, /api/analyze, /api/benchmark)
│   └── __init__.py           # Application factory
├── checkpoints/
│   └── best_model.pth        # Pre-trained Siamese U-Net weights
├── ml/
│   ├── dataset.py            # Bi-temporal PyTorch Dataset with data augmentation
│   ├── evaluate.py           # Precision, Recall, F1, and IoU evaluation suite
│   ├── loss.py               # Combined BCEDiceLoss for unbalanced change masks
│   ├── model.py              # PyTorch SiameseUNetDiff architecture definition
│   └── train.py              # Model training loop with checkpointing and validation
├── samples/                  # Pre-loaded baseline and target satellite demonstration pair
├── tests/                    # Pytest test suite (18 automated tests)
├── generate_samples.py       # Synthetic satellite pair generator
├── pytest.ini                # Pytest configuration
├── requirements.txt          # Python dependencies
├── run.py                    # Server startup script
└── README.md                 # Project documentation
```

---

## 💡 Technologies Used

* **Machine Learning / Computer Vision**: PyTorch, Torchvision, OpenCV, Scikit-Image, NumPy
* **Backend**: Python 3, Flask, Werkzeug
* **Frontend**: Vanilla JavaScript (ES6+), Modern Semantic HTML5, Custom Vanilla CSS (Design Tokens, Flexbox, CSS Grid)
* **Testing & Quality Assurance**: Pytest, Requests

---

## 👤 Author

Developed by [Adi Saha](https://github.com/Adi-Saha-07).
Contributions, feedback, and issue reports are welcome!
