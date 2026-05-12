<p align="center">
  <h1 align="center">Smart Parking Detection System</h1>
  <p align="center">A Dual-Model YOLO Framework for Real-Time Parking Space Detection</p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.11-red?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/Ultralytics-8.4-blue?logo=yolo" alt="Ultralytics">
  <img src="https://img.shields.io/badge/Flask-3.0-green?logo=flask" alt="Flask">
  <img src="https://img.shields.io/badge/Python-3.11-yellow?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="License">
  <img src="https://img.shields.io/badge/mAP50-0.994-brightgreen" alt="mAP50">
</p>

## Overview

This project presents a dual-model deep learning framework for real-time parking space occupancy detection. The system combines two YOLO architectures to handle both fixed-camera and arbitrary-viewpoint scenarios:

- **Specialized Model (YOLOv8n)**: Optimized for fixed CV Project camera views with 396 predefined Regions of Interest (ROIs) and Hungarian matching for high-precision detection
- **General Model (YOLO11n)**: Handles arbitrary parking lot viewpoints through direct YOLO detection, ensuring broad generalization

A view detection module automatically selects the appropriate model using object-count thresholding and ORB feature verification.

## Architecture

```
                    ┌─────────────┐
                    │  User Image  │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ View Detect │
                    │ (count+ORB) │
                    └──┬──────┬───┘
                       │      │
              CV View  │      │  General View
                       │      │
              ┌────────▼──┐ ┌─▼──────────┐
              │ YOLOv8n   │ │ YOLO11n    │
              │ + 396 ROI │ │ Direct     │
              │ + Hungarian│ │ Detection  │
              └─────┬─────┘ └──────┬─────┘
                    │              │
                    └──────┬───────┘
                           │
                    ┌──────▼──────┐
                    │  Annotated  │
                    │   Result    │
                    └─────────────┘
```

## Model Performance

| Model | Architecture | Params | GFLOPs | mAP50 | mAP50-95 | Precision | Recall |
|-------|-------------|--------|--------|-------|----------|-----------|--------|
| Specialized | YOLOv8n | 3.0M | 8.2 | **0.9944** | **0.9432** | 0.9972 | 0.9970 |
| General | YOLO11n | 2.6M | 6.4 | **0.9944** | **0.9251** | 0.9983 | 0.9983 |

## Experiment Results

| Metric | Original Paper | PKLot Fine-tuned | Improvement |
|--------|---------------|------------------|-------------|
| Training Images | 2,850 | **12,142** | 4.3× |
| Accuracy (mAP50) | 94% | **99.44%** | +5.4pp |
| Precision | 92% | **99.7%** | +7.7pp |
| Recall | 93% | **99.7%** | +6.7pp |
| mAP50-95 | — | **0.943** | — |
| FPS | 42 (24ms) | Same architecture | — |

> Full experiment report with training curves, confusion matrices, and analysis: [`results/experiment_report.docx`](results/experiment_report.docx)

## Project Structure

```
├── paper/                          # Research paper
│   ├── Smart_Parking_Paper.docx
│   └── Smart_Parking_Paper.pdf
├── backend/                        # Flask API + detection server
│   ├── app_final.py                # Main server (run this)
│   ├── train.py                    # YOLO training script
│   ├── run_experiment.py           # Dual-model experiment runner
│   ├── evaluate.py                 # Comprehensive evaluation suite
│   ├── export_models.py            # ONNX/TensorRT model export
│   ├── config.yaml                 # System configuration
│   ├── requirements.txt            # Python dependencies
│   └── parking_mask.png            # ROI mask (396 spaces)
├── frontend/                       # Interactive web UI
│   └── index.html                  # Drag-drop detection interface
├── models/                         # Trained model weights
│   ├── specialized_best.pt         # YOLOv8n fine-tuned on PKLot
│   └── general_best.pt             # YOLO11n fine-tuned on PKLot
├── results/                        # Experiment results
│   ├── experiment_report.docx      # Full Word report
│   ├── experiment_report.html      # HTML report
│   ├── specialized/                # Training curves, confusion matrix, metrics
│   └── general/                    # Training curves, confusion matrix, metrics
├── dataset.yaml.example            # Dataset config template
├── start_server.bat                # One-click server launcher
├── .gitignore
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (recommended)
- [PKLot dataset](https://web.inf.ufpr.br/vri/databases/parking-lot-database/) (download separately)

### Installation

```bash
# Clone the repository
git clone https://github.com/2499146834-arch/Smart-Parking-Detection.git
cd Smart-Parking-Detection

# Install dependencies
pip install -r backend/requirements.txt
```

### Dataset Setup

Download PKLot and organize as follows:

```
parking_dataset_real/
├── dataset.yaml
├── images/
│   ├── train/    (8,503 images)
│   ├── val/      (1,823 images)
│   └── test/     (1,816 images)
└── labels/
    ├── train/
    ├── val/
    └── test/
```

### Launch

**Windows:**
```bash
# Double-click start_server.bat
# OR
cd backend
python app_final.py
```

Then open **http://localhost:8000** in your browser.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Interactive detection frontend |
| `GET` | `/api/health` | Server health check + ROI count |
| `GET` | `/api/classes` | Available detection classes |
| `POST` | `/api/detect/image` | Upload image → annotated result + detection list |
| `POST` | `/api/detect/video` | Upload video → processed output |

### Detection Response Format

```json
{
  "detections": [
    { "id": 1, "class": "available", "confidence": 0.0, "bbox": {...} },
    { "id": 2, "class": "full", "confidence": 0.95, "bbox": {...} }
  ],
  "count": 396,
  "occupied": 312,
  "empty": 84,
  "image": "<base64>",
  "model_used": "CV Project YOLOv8n (ROI精准检测)",
  "view_type": "specialized",
  "inference_time": 0.234
}
```

## Training

```bash
cd backend

# Single model training
python train.py --data ../parking_dataset_real/dataset.yaml --model yolov8n.pt --epochs 30

# Dual-model experiment
python run_experiment.py
```

### Training Hyperparameters

| Parameter | Specialized (YOLOv8n) | General (YOLO11n) |
|-----------|----------------------|-------------------|
| Epochs | 30 | 30 |
| Image Size | 640 | 640 |
| Batch Size | 16 | 12 |
| Optimizer | AdamW | AdamW |
| LR Schedule | Cosine Annealing | Cosine Annealing |
| Augmentation | mosaic, mixup, HSV | mosaic, mixup, HSV |
| Close-mosaic | Last 10 epochs | Last 10 epochs |

### Why close-mosaic?

Mosaic augmentation improves generalization early in training, but can harm localization precision on real-scale images. By disabling mosaic for the last 10 epochs, the model fine-tunes on real-scale images, leading to a **40% improvement in mAP50-95** (0.67 → 0.94).

## Citation

```bibtex
@article{shi2026smartparking,
  title={A Dual-Model Deep Learning Framework for Real-Time Parking Space Detection:
         Architecture, Implementation, and Comprehensive Evaluation},
  author={Shi, Yunfeng and Ouyang, Qijing and Yang, Peilun and Pang, Jieyao and
          Su, Dinghang and Yang, Sijun and Feng, Qijing},
  journal={MScDS, Lingnan University},
  year={2026}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
