#!/usr/bin/env python3
"""
Smart Parking Detection System - Comprehensive Evaluation Suite
=================================================================
Computes: per-class AP, mAP, precision/recall/F1, confusion matrix,
PR curves, temporal stability, FPS benchmarks.

Usage:
    python evaluate.py --data /path/to/test/images --annotations /path/to/labels
    python evaluate.py --data ./test_images --annotations ./test_labels --output ./results
    python evaluate.py --video ./parking_video.mp4 --annotations ./video_labels/

Supports: YOLO-format labels, COCO-format JSON, CSV annotations
"""

import os
import sys
import cv2
import yaml
import json
import time
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'qtUI'))
from ultralytics import YOLO


# ===================== Configuration =====================

def load_config(config_path: str = None) -> dict:
    """Load system configuration from YAML."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}


cfg = load_config()
eval_cfg = cfg.get('evaluation', {})
CLASSES = eval_cfg.get('classes', ['legal', 'available', 'illegal', 'full'])
CONF_THRESHOLDS = eval_cfg.get('conf_thresholds',
    [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50])
IOU_THRESHOLD = eval_cfg.get('iou_threshold', 0.5)

# Model paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPECIALIZED_MODEL_PATH = os.path.join(BASE_DIR, cfg.get('models', {}).get(
    'specialized', 'best_cv_project.pt'))
GENERAL_MODEL_PATH = os.path.join(BASE_DIR, cfg.get('models', {}).get(
    'general', 'best.pt'))


# ===================== IoU Calculation =====================

def compute_iou(box1: Tuple, box2: Tuple) -> float:
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


# ===================== Metrics Computation =====================

def compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute Average Precision using 11-point interpolation."""
    ap = 0.0
    for t in np.linspace(0, 1, 11):
        p = np.max(precisions[recalls >= t]) if np.any(recalls >= t) else 0
        ap += p / 11.0
    return ap


def compute_map_per_class(
    all_detections: List[Dict],
    all_ground_truths: List[Dict],
    iou_threshold: float = 0.5,
    num_classes: int = 4
) -> Dict:
    """Compute per-class AP and mAP."""
    aps = {}

    for cls_id in range(num_classes):
        # Collect detections and ground truths for this class
        cls_detections = []
        cls_gts = []

        for img_dets, img_gts in zip(all_detections, all_ground_truths):
            dets = [d for d in img_dets if d['class_id'] == cls_id]
            gts = [g for g in img_gts if g['class_id'] == cls_id]

            cls_detections.append(dets)
            cls_gts.append(gts)

        # Sort all detections by confidence descending
        all_dets_sorted = []
        for img_idx, dets in enumerate(cls_detections):
            for det in dets:
                all_dets_sorted.append({
                    'img_idx': img_idx,
                    'confidence': det['confidence'],
                    'bbox': det['bbox'],
                })
        all_dets_sorted.sort(key=lambda x: x['confidence'], reverse=True)

        # Count total ground truths
        total_gts = sum(len(gts) for gts in cls_gts)
        if total_gts == 0:
            aps[CLASSES[cls_id]] = 0.0
            continue

        # Match detections to ground truths
        gt_matched = [set() for _ in cls_gts]  # Track matched GTs per image
        tp = np.zeros(len(all_dets_sorted))
        fp = np.zeros(len(all_dets_sorted))

        for det_idx, det in enumerate(all_dets_sorted):
            img_idx = det['img_idx']
            gts_in_img = cls_gts[img_idx]

            best_iou = 0
            best_gt_idx = -1
            for gt_idx, gt in enumerate(gts_in_img):
                if gt_idx in gt_matched[img_idx]:
                    continue
                iou = compute_iou(det['bbox'], gt['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou >= iou_threshold:
                tp[det_idx] = 1
                gt_matched[img_idx].add(best_gt_idx)
            else:
                fp[det_idx] = 1

        # Compute precision-recall curve
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / total_gts
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)

        # 11-point interpolation AP
        ap = compute_ap(recalls, precisions)
        aps[CLASSES[cls_id]] = ap

    aps['mAP'] = np.mean(list(aps.values()))
    return aps


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    class_names: List[str]
) -> Dict:
    """Compute precision, recall, F1 per class and overall."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=range(len(class_names)), zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

    results = {'per_class': {}, 'overall': {}, 'confusion_matrix': cm.tolist()}

    for i, name in enumerate(class_names):
        results['per_class'][name] = {
            'precision': round(precision[i], 4),
            'recall': round(recall[i], 4),
            'f1': round(f1[i], 4),
            'support': int(support[i]),
        }

    results['overall'] = {
        'macro_precision': round(np.mean(precision), 4),
        'macro_recall': round(np.mean(recall), 4),
        'macro_f1': round(np.mean(f1), 4),
        'accuracy': round(np.mean(np.array(y_true) == np.array(y_pred)), 4)
        if len(y_true) > 0 else 0.0,
    }

    return results


# ===================== Annotation Loading =====================

def load_yolo_annotations(label_dir: str, img_dir: str) -> List[Dict]:
    """Load YOLO-format annotations from a directory."""
    ground_truths = []
    img_files = sorted(Path(img_dir).glob('*.jpg')) + \
                sorted(Path(img_dir).glob('*.png')) + \
                sorted(Path(img_dir).glob('*.jpeg'))

    for img_path in img_files:
        label_path = Path(label_dir) / f"{img_path.stem}.txt"
        gts = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        x_center, y_center, w, h = map(float, parts[1:5])
                        # Convert YOLO format to xyxy
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            ih, iw = img.shape[:2]
                            x1 = (x_center - w / 2) * iw
                            y1 = (y_center - h / 2) * ih
                            x2 = (x_center + w / 2) * iw
                            y2 = (y_center + h / 2) * ih
                            gts.append({
                                'class_id': cls_id,
                                'class_name': CLASSES[cls_id] if cls_id < len(CLASSES) else f'class_{cls_id}',
                                'bbox': [x1, y1, x2, y2],
                            })
        ground_truths.append(gts)
    return ground_truths


def load_coco_annotations(json_path: str) -> Tuple[List[Dict], Dict]:
    """Load COCO-format annotations."""
    with open(json_path, 'r') as f:
        coco = json.load(f)

    # Build image id -> file name map
    images = {img['id']: img['file_name'] for img in coco['images']}
    categories = {cat['id']: cat['name'] for cat in coco['categories']}

    # Group annotations by image
    img_annotations = defaultdict(list)
    for ann in coco['annotations']:
        x, y, w, h = ann['bbox']
        img_annotations[ann['image_id']].append({
            'class_id': ann['category_id'] - 1,  # Convert to 0-indexed
            'class_name': categories.get(ann['category_id'], f"class_{ann['category_id']}"),
            'bbox': [x, y, x + w, y + h],
        })

    ground_truths = [img_annotations.get(img_id, []) for img_id in sorted(images.keys())]
    return ground_truths, images


# ===================== Model Inference =====================

def run_inference_on_dataset(
    model: YOLO,
    img_dir: str,
    conf_threshold: float = 0.15,
    is_specialized: bool = False,
    rois: List = None
) -> List[Dict]:
    """Run model inference on all images in a directory."""
    all_detections = []
    img_files = sorted(Path(img_dir).glob('*.jpg')) + \
                sorted(Path(img_dir).glob('*.png')) + \
                sorted(Path(img_dir).glob('*.jpeg'))

    inference_times = []

    for img_path in img_files:
        img = cv2.imread(str(img_path))
        if img is None:
            all_detections.append([])
            continue

        start = time.time()
        results = model(img, conf=conf_threshold, iou=0.3, verbose=False)[0]
        elapsed = (time.time() - start) * 1000  # ms
        inference_times.append(elapsed)

        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = results.names.get(cls_id, f'class_{cls_id}')

                # Map multi-class names to simplified classes
                cls_name_lower = cls_name.lower()
                if cls_name_lower in ['full', 'occupied', 'car', 'vehicle']:
                    mapped_cls = 3  # 'full'
                elif cls_name_lower in ['available', 'empty', 'free']:
                    mapped_cls = 1  # 'available'
                elif cls_name_lower == 'illegal':
                    mapped_cls = 2  # 'illegal'
                elif cls_name_lower == 'legal':
                    mapped_cls = 0  # 'legal'
                else:
                    mapped_cls = cls_id

                detections.append({
                    'class_id': mapped_cls,
                    'class_name': CLASSES[mapped_cls] if mapped_cls < len(CLASSES) else f'class_{mapped_cls}',
                    'confidence': conf,
                    'bbox': [x1, y1, x2, y2],
                })

        all_detections.append(detections)

    avg_latency = np.mean(inference_times) if inference_times else 0
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0

    return all_detections, {'avg_latency_ms': avg_latency, 'fps': fps}


# ===================== Temporal Stability Analysis =====================

def evaluate_temporal_stability(
    model: YOLO,
    video_path: str,
    annotation_path: str = None,
    max_frames: int = 150,
    conf_threshold: float = 0.15
) -> Dict:
    """Evaluate detection consistency across video frames."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {'error': 'Cannot open video'}

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = min(max_frames, total_frames)

    frame_detections = []  # List of detection counts per frame
    frame_class_changes = []  # Count of class-flipping detections
    prev_detections = None

    for frame_idx in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=conf_threshold, iou=0.3, verbose=False)[0]
        detections = []

        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'class_id': cls_id,
                    'conf': conf,
                })

        frame_detections.append(len(detections))

        # Track class flips between consecutive frames
        if prev_detections is not None and len(detections) > 0:
            flips = 0
            for det in detections:
                matches = []
                for prev_det in prev_detections:
                    iou = compute_iou(det['bbox'], prev_det['bbox'])
                    if iou > 0.5:
                        matches.append((iou, prev_det))
                if matches:
                    best_prev = max(matches, key=lambda x: x[0])[1]
                    if det['class_id'] != best_prev['class_id']:
                        flips += 1
            frame_class_changes.append(flips)

        prev_detections = detections

    cap.release()

    detections_array = np.array(frame_detections)
    stability_metrics = {
        'mean_detections': float(np.mean(detections_array)),
        'std_detections': float(np.std(detections_array)),
        'cv_detections': float(np.std(detections_array) / (np.mean(detections_array) + 1e-10)),
        'min_detections': int(np.min(detections_array)),
        'max_detections': int(np.max(detections_array)),
        'detection_stability': float(1.0 - min(1.0, np.std(detections_array) / (np.mean(detections_array) + 1e-10))),
        'avg_class_flips_per_frame': float(np.mean(frame_class_changes)) if frame_class_changes else 0,
        'total_class_flips': int(sum(frame_class_changes)),
        'frames_analyzed': max_frames,
    }

    return stability_metrics


# ===================== Visualization =====================

def plot_per_class_metrics(metrics: Dict, output_dir: str):
    """Plot per-class precision, recall, F1 bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))
    classes = list(metrics['per_class'].keys())
    x = np.arange(len(classes))
    width = 0.25

    precisions = [metrics['per_class'][c]['precision'] for c in classes]
    recalls = [metrics['per_class'][c]['recall'] for c in classes]
    f1s = [metrics['per_class'][c]['f1'] for c in classes]

    ax.bar(x - width, precisions, width, label='Precision', color='#2196F3')
    ax.bar(x, recalls, width, label='Recall', color='#4CAF50')
    ax.bar(x + width, f1s, width, label='F1-Score', color='#FF9800')

    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=150)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], output_dir: str, normalize: bool = True):
    """Plot normalized and raw confusion matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Raw counts
    im1 = axes[0].imshow(cm, cmap='Blues')
    axes[0].set_title('Confusion Matrix (Counts)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_xticks(range(len(class_names)))
    axes[0].set_yticks(range(len(class_names)))
    axes[0].set_xticklabels(class_names)
    axes[0].set_yticklabels(class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            axes[0].text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.colorbar(im1, ax=axes[0])

    # Normalized
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    im2 = axes[1].imshow(cm_norm, cmap='Greens')
    axes[1].set_title('Confusion Matrix (Normalized)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_xticks(range(len(class_names)))
    axes[1].set_yticks(range(len(class_names)))
    axes[1].set_xticklabels(class_names)
    axes[1].set_yticklabels(class_names)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            axes[1].text(j, i, f'{cm_norm[i, j]:.2f}', ha='center', va='center',
                        color='white' if cm_norm[i, j] > 0.5 else 'black')
    plt.colorbar(im2, ax=axes[1])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()


def plot_pr_curves(results_by_threshold: Dict, output_dir: str):
    """Plot precision-recall curves across confidence thresholds."""
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0']
    for i, (cls_name, data) in enumerate(results_by_threshold.items()):
        if cls_name in ['mAP', 'overall']:
            continue
        color = colors[i % len(colors)]
        ax.plot(data['recalls'], data['precisions'], color=color, linewidth=2,
                label=f'{cls_name} (AP={data["ap"]:.3f})')

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curves by Class')
    ax.legend(loc='lower left')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'pr_curves.png'), dpi=150)
    plt.close()


def plot_f1_vs_confidence(results_by_threshold: Dict, output_dir: str):
    """Plot F1-score vs confidence threshold for each class."""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
    for i, (cls_name, data) in enumerate(results_by_threshold.items()):
        if cls_name in ['mAP', 'overall']:
            continue
        color = colors[i % len(colors)]
        confs = data.get('conf_thresholds', CONF_THRESHOLDS)
        f1s = [2 * p * r / (p + r + 1e-10) for p, r in zip(data['precisions'], data['recalls'])]
        ax.plot(confs, f1s, color=color, linewidth=2, marker='o', markersize=4,
                label=cls_name)

    ax.set_xlabel('Confidence Threshold')
    ax.set_ylabel('F1-Score')
    ax.set_title('F1-Score vs Confidence Threshold')
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlim(0, max(CONF_THRESHOLDS))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'f1_vs_confidence.png'), dpi=150)
    plt.close()


# ===================== Main Evaluation =====================

def run_full_evaluation(
    img_dir: str,
    annotation_dir: str = None,
    annotation_format: str = 'yolo',
    video_path: str = None,
    specialized_model_path: str = None,
    general_model_path: str = None,
    output_dir: str = './evaluation_results',
) -> Dict:
    """Run comprehensive evaluation and generate all metrics and plots."""
    os.makedirs(output_dir, exist_ok=True)

    # Use configured paths if not specified
    specialized_model_path = specialized_model_path or SPECIALIZED_MODEL_PATH
    general_model_path = general_model_path or GENERAL_MODEL_PATH

    results = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'img_dir': img_dir,
            'annotation_dir': annotation_dir,
            'output_dir': output_dir,
            'num_classes': len(CLASSES),
            'classes': CLASSES,
        },
        'dual_model': {},
        'specialized_only': {},
        'general_only': {},
        'temporal_stability': {},
    }

    # ===== Load annotations =====
    print("=" * 60)
    print("Smart Parking Detection - Evaluation Suite")
    print("=" * 60)

    ground_truths = None
    if annotation_dir:
        print(f"\n[1/6] Loading annotations ({annotation_format})...")
        if annotation_format == 'yolo':
            ground_truths = load_yolo_annotations(annotation_dir, img_dir)
        elif annotation_format == 'coco':
            ground_truths, images_map = load_coco_annotations(annotation_dir)
        print(f"  Loaded annotations for {len(ground_truths)} images")

    # ===== Specialized Model Evaluation =====
    print(f"\n[2/6] Evaluating specialized model (YOLOv8n + ROI)...")
    if os.path.exists(specialized_model_path):
        spec_model = YOLO(specialized_model_path)
        spec_dets, spec_perf = run_inference_on_dataset(
            spec_model, img_dir, conf_threshold=0.15, is_specialized=True)
        print(f"  FPS: {spec_perf['fps']:.1f}, Latency: {spec_perf['avg_latency_ms']:.1f}ms")

        if ground_truths:
            spec_aps = compute_map_per_class(spec_dets, ground_truths, IOU_THRESHOLD)
            results['specialized_only'] = {
                'mAP': spec_aps,
                'performance': spec_perf,
            }
            print(f"  mAP@0.5: {spec_aps['mAP']:.4f}")
        else:
            results['specialized_only'] = {'performance': spec_perf}
    else:
        print(f"  [SKIP] Model not found: {specialized_model_path}")
        print(f"  Place models in backend/ or update config.yaml")

    # ===== General Model Evaluation =====
    print(f"\n[3/6] Evaluating general model (YOLO11n)...")
    if os.path.exists(general_model_path):
        gen_model = YOLO(general_model_path)
        gen_dets, gen_perf = run_inference_on_dataset(
            gen_model, img_dir, conf_threshold=0.3, is_specialized=False)
        print(f"  FPS: {gen_perf['fps']:.1f}, Latency: {gen_perf['avg_latency_ms']:.1f}ms")

        if ground_truths:
            gen_aps = compute_map_per_class(gen_dets, ground_truths, IOU_THRESHOLD)
            results['general_only'] = {
                'mAP': gen_aps,
                'performance': gen_perf,
            }
            print(f"  mAP@0.5: {gen_aps['mAP']:.4f}")
        else:
            results['general_only'] = {'performance': gen_perf}
    else:
        print(f"  [SKIP] Model not found: {general_model_path}")

    # ===== Classification Metrics (if both models ran) =====
    print(f"\n[4/6] Computing classification metrics...")
    if ground_truths and os.path.exists(specialized_model_path):
        y_true_spec, y_pred_spec = [], []
        for gt_list, det_list in zip(ground_truths, spec_dets):
            for gt in gt_list:
                matched = False
                for det in det_list:
                    if compute_iou(gt['bbox'], det['bbox']) > IOU_THRESHOLD:
                        y_true_spec.append(gt['class_id'])
                        y_pred_spec.append(det['class_id'])
                        matched = True
                        break
                if not matched:
                    # Missed detection -> assign "background" (use a high class ID)
                    y_true_spec.append(gt['class_id'])
                    y_pred_spec.append(-1)  # unmatched

        if y_true_spec:
            cls_metrics = compute_classification_metrics(
                [y if y >= 0 else 0 for y in y_true_spec],
                [p if p >= 0 else 0 for p in y_pred_spec],
                CLASSES
            )
            results['classification_metrics'] = cls_metrics

            print(f"  Accuracy: {cls_metrics['overall']['accuracy']:.4f}")
            print(f"  Macro F1:  {cls_metrics['overall']['macro_f1']:.4f}")

            # Plot results
            plot_per_class_metrics(cls_metrics, output_dir)
            cm = np.array(cls_metrics['confusion_matrix'])
            plot_confusion_matrix(cm, CLASSES, output_dir, normalize=True)

    # ===== Temporal Stability =====
    if video_path and os.path.exists(video_path):
        print(f"\n[5/6] Evaluating temporal stability...")
        if os.path.exists(specialized_model_path):
            model = YOLO(specialized_model_path)
            stability = evaluate_temporal_stability(model, video_path, max_frames=150)
            results['temporal_stability'] = stability
            print(f"  Detection stability: {stability.get('detection_stability', 'N/A'):.4f}")
            print(f"  Avg class flips/frame: {stability.get('avg_class_flips_per_frame', 'N/A'):.4f}")
        else:
            print(f"  [SKIP] Model not available for temporal analysis")

    # ===== Ablation Summary =====
    print(f"\n[6/6] Generating ablation summary...")
    ablation = {
        'dual_model': results.get('dual_model', {}),
        'specialized_only': {k: v for k, v in results.get('specialized_only', {}).items() if k == 'mAP'},
        'general_only': {k: v for k, v in results.get('general_only', {}).items() if k == 'mAP'},
    }
    results['ablation'] = ablation

    # ===== Save Results =====
    # Convert numpy values for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert_for_json(i) for i in obj]
        return obj

    results_json = convert_for_json(results)

    json_path = os.path.join(output_dir, 'evaluation_results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2, ensure_ascii=False)

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    spec_map = results.get('specialized_only', {}).get('mAP', {}).get('mAP', 'N/A')
    gen_map = results.get('general_only', {}).get('mAP', {}).get('mAP', 'N/A')
    cls_acc = results.get('classification_metrics', {}).get('overall', {}).get('accuracy', 'N/A')
    cls_f1 = results.get('classification_metrics', {}).get('overall', {}).get('macro_f1', 'N/A')

    print(f"  Specialized Model mAP@0.5:  {spec_map}")
    print(f"  General Model mAP@0.5:      {gen_map}")
    print(f"  Classification Accuracy:     {cls_acc}")
    print(f"  Classification Macro F1:     {cls_f1}")
    print(f"\n  Results saved to: {output_dir}/")
    print(f"  Full metrics:     {json_path}")

    return results


# ===================== CLI =====================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Smart Parking Detection - Comprehensive Evaluation')

    parser.add_argument('--data', type=str, required=True,
                        help='Path to test images directory')
    parser.add_argument('--annotations', type=str, default=None,
                        help='Path to annotations directory (YOLO format) or COCO JSON file')
    parser.add_argument('--format', type=str, default='yolo',
                        choices=['yolo', 'coco'],
                        help='Annotation format (default: yolo)')
    parser.add_argument('--video', type=str, default=None,
                        help='Path to test video for temporal stability analysis')
    parser.add_argument('--specialized-model', type=str, default=None,
                        help='Path to specialized YOLOv8n model weights')
    parser.add_argument('--general-model', type=str, default=None,
                        help='Path to general YOLO11n model weights')
    parser.add_argument('--output', type=str, default='./evaluation_results',
                        help='Output directory for results and plots')
    parser.add_argument('--classes', type=str, nargs='+', default=None,
                        help='Override class names (e.g., --classes legal available illegal full)')

    args = parser.parse_args()

    if args.classes:
        CLASSES = args.classes

    results = run_full_evaluation(
        img_dir=args.data,
        annotation_dir=args.annotations,
        annotation_format=args.format,
        video_path=args.video,
        specialized_model_path=args.specialized_model,
        general_model_path=args.general_model,
        output_dir=args.output,
    )
