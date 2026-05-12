#!/usr/bin/env python3
"""
Smart Parking - Model Export & Optimization
============================================
Exports YOLO models to ONNX, TensorRT, and OpenVINO formats.
Includes INT8 quantization and inference benchmarking.

Usage:
    python export_models.py --model best_cv_project.pt --formats onnx
    python export_models.py --model best.pt --all
    python export_models.py --model best.pt --benchmark
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'qtUI'))
from ultralytics import YOLO


def export_to_onnx(model_path, output_dir=None, simplify=True):
    if output_dir is None:
        output_dir = os.path.dirname(model_path) or '.'
    model = YOLO(model_path)
    model_name = Path(model_path).stem
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")

    print(f"\n[ONNX] Exporting {model_path} -> {onnx_path}")
    model.export(format='onnx', imgsz=640, simplify=simplify, opset=12, dynamic=False)

    size_mb = os.path.getsize(onnx_path) / (1024 * 1024) if os.path.exists(onnx_path) else 0
    print(f"  ONNX model: {size_mb:.1f} MB")
    return onnx_path if os.path.exists(onnx_path) else None


def export_to_tensorrt(model_path, output_dir=None, fp16=True):
    if output_dir is None:
        output_dir = os.path.dirname(model_path) or '.'
    model = YOLO(model_path)
    model_name = Path(model_path).stem
    engine_path = os.path.join(output_dir, f"{model_name}.engine")

    print(f"\n[TensorRT] Exporting {model_path} -> {engine_path}")
    try:
        model.export(format='engine', imgsz=640, half=fp16, workspace=4, dynamic=False)
        size_mb = os.path.getsize(engine_path) / (1024 * 1024) if os.path.exists(engine_path) else 0
        print(f"  TensorRT engine: {size_mb:.1f} MB")
        return engine_path if os.path.exists(engine_path) else None
    except Exception as e:
        print(f"  [SKIP] TensorRT export failed: {e}")
        return None


def export_to_tflite(model_path, output_dir=None, int8=False):
    if output_dir is None:
        output_dir = os.path.dirname(model_path) or '.'
    model = YOLO(model_path)
    model_name = Path(model_path).stem
    tflite_path = os.path.join(output_dir, f"{model_name}_float16.tflite")

    print(f"\n[TFLite] Exporting {model_path} -> {tflite_path}")
    try:
        model.export(format='tflite', imgsz=640, half=True, int8=int8)
        size_mb = os.path.getsize(tflite_path) / (1024 * 1024) if os.path.exists(tflite_path) else 0
        print(f"  TFLite model: {size_mb:.1f} MB")
        return tflite_path if os.path.exists(tflite_path) else None
    except Exception as e:
        print(f"  [SKIP] TFLite export failed: {e}")
        return None


def benchmark_model(model_path, num_runs=100, resolutions=None, device='cpu'):
    if resolutions is None:
        resolutions = [(640, 480), (1280, 720), (1920, 1080)]

    model = YOLO(model_path)
    model.to(device)

    results = {'model': os.path.basename(model_path), 'device': device, 'resolutions': {}}

    print(f"\n[Benchmark] {model_path} on {device}")
    print(f"{'Resolution':<16} {'FPS':>8} {'Latency(ms)':>14}")
    print("-" * 42)

    for w, h in resolutions:
        test_img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        for _ in range(10):
            _ = model(test_img, verbose=False)

        latencies = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(test_img, verbose=False)
            latencies.append((time.perf_counter() - start) * 1000)

        avg_latency = np.mean(latencies)
        fps = 1000.0 / avg_latency
        results['resolutions'][f'{w}x{h}'] = {
            'fps': round(fps, 1),
            'latency_ms': round(avg_latency, 1),
            'latency_p95_ms': round(np.percentile(latencies, 95), 1),
        }
        print(f"{w}x{h:<12} {fps:>8.1f} {avg_latency:>12.1f}")

    return results


def compare_formats(model_path):
    """Compare PyTorch vs ONNX Runtime speed."""
    import onnxruntime as ort

    print("\n[Compare] PyTorch vs ONNX Runtime")

    # PyTorch benchmark
    pt_results = benchmark_model(model_path, num_runs=50)
    pt_fps_1080 = pt_results['resolutions']['1920x1080']['fps']

    # ONNX benchmark
    onnx_path = export_to_onnx(model_path)
    if onnx_path:
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        test_input = np.random.rand(1, 3, 640, 640).astype(np.float32)

        for _ in range(10):
            _ = session.run(None, {input_name: test_input})

        latencies = []
        for _ in range(50):
            start = time.perf_counter()
            _ = session.run(None, {input_name: test_input})
            latencies.append((time.perf_counter() - start) * 1000)

        onnx_fps = 1000.0 / np.mean(latencies)
        speedup = onnx_fps / pt_fps_1080 if pt_fps_1080 > 0 else 0
        print(f"\n  PyTorch FPS: {pt_fps_1080:.1f} -> ONNX FPS: {onnx_fps:.1f} ({speedup:.2f}x speedup)")

    return {'pytorch': pt_results, 'onnx_fps': round(onnx_fps, 1), 'speedup': round(speedup, 2)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLO Model Export & Benchmark')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLO .pt model')
    parser.add_argument('--formats', type=str, nargs='+', default=['onnx'],
                        choices=['onnx', 'engine', 'tflite'])
    parser.add_argument('--all', action='store_true', help='Export all supported formats')
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--benchmark', action='store_true')
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', '0'])

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Error: Model not found: {args.model}")
        sys.exit(1)

    export_handlers = {'onnx': export_to_onnx, 'engine': export_to_tensorrt, 'tflite': export_to_tflite}

    if args.all:
        args.formats = ['onnx', 'engine', 'tflite']

    exported = {}
    for fmt in args.formats:
        exported[fmt] = export_handlers[fmt](args.model, args.output)

    if args.benchmark:
        benchmark_model(args.model, device=args.device)

    if args.compare:
        compare_formats(args.model)

    # Summary
    print("\n" + "=" * 50)
    print("Export Summary:")
    orig_mb = os.path.getsize(args.model) / (1024 * 1024)
    print(f"  Original: {orig_mb:.1f} MB")
    for fmt, path in exported.items():
        if path and os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            reduction = (1 - size_mb / orig_mb) * 100
            print(f"  {fmt:12s} {size_mb:.1f} MB ({reduction:+.0f}%)")
        else:
            print(f"  {fmt:12s} [FAILED]")
