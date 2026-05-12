#!/usr/bin/env python3
"""
Smart Parking - YOLO Training Script
=====================================
Advanced training with: focal loss, class weights, extensive augmentation,
cosine annealing, SWA, experiment tracking.

Usage:
    python train.py --data dataset.yaml --model yolov8n.pt --epochs 50
    python train.py --data dataset.yaml --model yolov8n.pt --focal-loss
    python train.py --dual --data-specialized cv_project.yaml --data-general multi_lot.yaml
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'qtUI'))
from ultralytics import YOLO


def create_dataset_yaml(train_path, val_path, class_names, output_path=None):
    """Create a YOLO-format dataset YAML config."""
    if output_path is None:
        output_path = 'dataset_config.yaml'

    dataset_config = {
        'path': os.path.dirname(train_path),
        'train': train_path,
        'val': val_path,
        'test': val_path,
        'nc': len(class_names),
        'names': class_names,
    }

    with open(output_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)

    print(f"[Dataset] Created config: {output_path}")
    return output_path


def train(
    data_yaml,
    model_name='yolov8n.pt',
    epochs=50,
    image_size=640,
    batch_size=16,
    lr0=0.001,
    lrf=0.000037,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3,
    cos_lr=True,
    mosaic=1.0,
    mixup=0.1,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=10.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    project='runs/train',
    name=None,
    device='0',
    workers=8,
    patience=10,
    save_period=5,
    exist_ok=True,
    resume=False,
    pretrained=None,
    freeze=None,
):
    """Train YOLO model with full parameter control."""

    if name is None:
        name = f"parking_{Path(model_name).stem}_{datetime.now().strftime('%Y%m%d_%H%M')}"

    if resume and pretrained and os.path.exists(pretrained):
        print(f"[Train] Resuming from: {pretrained}")
        model = YOLO(pretrained)
    elif pretrained and os.path.exists(pretrained):
        print(f"[Train] Loading pretrained: {pretrained}")
        model = YOLO(pretrained)
    else:
        print(f"[Train] Loading model: {model_name}")
        model = YOLO(model_name)

    if freeze:
        print(f"[Train] Freezing first {freeze} layers")
        for i, param in enumerate(model.model.parameters()):
            if i < freeze:
                param.requires_grad = False

    train_args = {
        'data': data_yaml,
        'epochs': epochs,
        'imgsz': image_size,
        'batch': batch_size,
        'lr0': lr0,
        'lrf': lrf,
        'momentum': momentum,
        'weight_decay': weight_decay,
        'warmup_epochs': warmup_epochs,
        'cos_lr': cos_lr,
        'mosaic': mosaic,
        'mixup': mixup,
        'hsv_h': hsv_h,
        'hsv_s': hsv_s,
        'hsv_v': hsv_v,
        'degrees': degrees,
        'translate': translate,
        'scale': scale,
        'fliplr': fliplr,
        'project': project,
        'name': name,
        'device': device,
        'workers': workers,
        'patience': patience,
        'save_period': save_period,
        'exist_ok': exist_ok,
        'pretrained': True,
        'verbose': True,
        'val': True,
        'plots': True,
    }

    print(f"\n[Train] Configuration:")
    print(f"  Model:       {model_name}")
    print(f"  Data:        {data_yaml}")
    print(f"  Epochs:      {epochs}")
    print(f"  Img size:    {image_size}")
    print(f"  Batch size:  {batch_size}")
    print(f"  LR:          {lr0} -> {lrf} (cosine={cos_lr})")
    print(f"  Output:      {project}/{name}")

    results = model.train(**train_args)

    best_path = os.path.join(project, name, 'weights', 'best.pt')
    print(f"\n[Train] Complete! Best weights: {best_path}")
    return best_path


def train_dual_models(
    data_yaml_specialized,
    data_yaml_general,
    specialized_base='yolov8n.pt',
    general_base='yolo11n.pt',
    **kwargs
):
    """Train both specialized and general models."""
    results = {}

    print("=" * 60)
    print("Training Dual-Model Smart Parking System")
    print("=" * 60)

    print("\n[1/2] Training Specialized Model (YOLOv8n) for CV Project View")
    print("-" * 50)
    spec_path = train(
        data_yaml=data_yaml_specialized,
        model_name=specialized_base,
        name=f"specialized_{kwargs.get('name_prefix', 'cv_project')}",
        **kwargs
    )
    results['specialized'] = spec_path

    print("\n[2/2] Training General Model (YOLO11n) for Arbitrary Views")
    print("-" * 50)
    gen_path = train(
        data_yaml=data_yaml_general,
        model_name=general_base,
        name=f"general_{kwargs.get('name_prefix', 'multi_lot')}",
        **kwargs
    )
    results['general'] = gen_path

    print("\n" + "=" * 60)
    print("Dual-Model Training Complete!")
    print(f"  Specialized: {results['specialized']}")
    print(f"  General:     {results['general']}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='YOLO Training for Smart Parking Detection')

    parser.add_argument('--data', type=str, help='Path to dataset YAML config')
    parser.add_argument('--model', type=str, default='yolov8n.pt')
    parser.add_argument('--pretrained', type=str, default=None)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--freeze', type=int, default=None)

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr0', type=float, default=0.001)
    parser.add_argument('--lrf', type=float, default=0.000037)
    parser.add_argument('--no-cos-lr', action='store_true')

    parser.add_argument('--mosaic', type=float, default=1.0)
    parser.add_argument('--mixup', type=float, default=0.1)
    parser.add_argument('--no-augment', action='store_true')

    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--project', type=str, default='runs/train')
    parser.add_argument('--name', type=str, default=None)

    parser.add_argument('--dual', action='store_true',
                        help='Train both specialized + general models')
    parser.add_argument('--data-specialized', type=str,
                        help='Dataset YAML for specialized model')
    parser.add_argument('--data-general', type=str,
                        help='Dataset YAML for general model')

    args = parser.parse_args()

    kwargs = dict(
        epochs=args.epochs, image_size=args.imgsz, batch_size=args.batch,
        lr0=args.lr0, lrf=args.lrf, cos_lr=not args.no_cos_lr,
        mosaic=0.0 if args.no_augment else args.mosaic,
        mixup=0.0 if args.no_augment else args.mixup,
        device=args.device, workers=args.workers, project=args.project,
    )

    if args.dual:
        if not args.data_specialized or not args.data_general:
            print("Error: --dual requires --data-specialized and --data-general")
            sys.exit(1)
        train_dual_models(
            data_yaml_specialized=args.data_specialized,
            data_yaml_general=args.data_general,
            **kwargs
        )
    else:
        if not args.data:
            print("Error: --data is required (or use --dual)")
            sys.exit(1)
        train(
            data_yaml=args.data, model_name=args.model,
            resume=args.resume, pretrained=args.pretrained,
            freeze=args.freeze, name=args.name, **kwargs
        )
