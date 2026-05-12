# Smart Parking Detection System

A Dual-Model Deep Learning Framework for Real-Time Parking Space Detection.

## Architecture

| Model | Architecture | Parameters | mAP50 | mAP50-95 | Usage |
|-------|-------------|-----------|-------|----------|-------|
| Specialized | YOLOv8n | 3.0M | 0.994 | 0.943 | Fixed CV Project camera + 396 ROIs |
| General | YOLO11n | 2.6M | 0.994 | 0.925 | Arbitrary viewpoints |

## Project Structure

```
├── paper/                  # Research paper
├── backend/                # Flask API + detection server
│   ├── app_final.py        # Main server (run this)
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation suite
│   ├── config.yaml         # Configuration
│   └── parking_mask.png    # ROI mask (396 parking spaces)
├── frontend/               # Web UI
│   └── index.html
├── models/                 # Trained weights
│   ├── specialized_best.pt
│   └── general_best.pt
├── results/                # Experiment results & reports
│   ├── specialized/        # Specialized model training curves
│   ├── general/            # General model training curves
│   └── experiment_report.* # Full experiment report
└── dataset.yaml.example    # Dataset config template
```

## Quick Start

```bash
# Install dependencies
pip install -r backend/requirements.txt

# Start server
cd backend
python app_final.py

# Open http://localhost:8000
```

## Dataset

Fine-tuned on [PKLot](https://web.inf.ufpr.br/vri/databases/parking-lot-database/) (12,142 images, 2 classes: empty / occupied).

Prepare dataset structure:
```
parking_dataset_real/
├── dataset.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

## Training

```bash
cd backend
python train.py --data ../parking_dataset_real/dataset.yaml --model yolov8n.pt --epochs 30
```

## API

- `POST /api/detect/image` — Upload image, returns annotated result with detection list
- `GET /api/health` — Server health check

## Results

| Metric | Old (paper) | New (PKLot fine-tuned) | Improvement |
|--------|------------|----------------------|-------------|
| Dataset | 2,850 images | 12,142 images | 4.3x |
| mAP50 | ~94% | 99.44% | +5.4pp |
| mAP50-95 | — | 0.943 | — |
| Precision | 92% | 99.7% | +7.7pp |
| Recall | 93% | 99.7% | +6.7pp |

## License

MIT
