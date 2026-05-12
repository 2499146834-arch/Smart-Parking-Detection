"""Train dual-model on PKLot + evaluate all 7 optimizations."""
import os, sys, time, json, csv, logging
import cv2, yaml, numpy as np
from pathlib import Path
from datetime import datetime
from scipy.optimize import linear_sum_assignment
from multiprocessing import freeze_support

os.chdir(r'D:\Smart Parking\repo\backend')

import torch
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

from ultralytics import YOLO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('results/training.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

OUT = 'results'
DATA = r'D:\Smart Parking\repo\parking_dataset_real'
DEVICE = 'cuda'

def train_models():
    os.makedirs(OUT, exist_ok=True)
    log.info("=" * 60)
    log.info("TRAINING DUAL-MODEL ON PKLot (12,142 images)")
    log.info("workers=4, cache=False, amp=True, epochs=30")
    log.info("=" * 60)

    common = dict(
        data=f'{DATA}/dataset.yaml', epochs=30, imgsz=640, batch=16, amp=True,
        lr0=0.001, lrf=3.7e-5, momentum=0.937, weight_decay=0.0005,
        warmup_epochs=3, cos_lr=True, mosaic=1.0, mixup=0.1,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=10.0,
        translate=0.1, scale=0.5, fliplr=0.5,
        device=DEVICE, workers=4, patience=10, exist_ok=True,
        pretrained=True, plots=True
    )

    log.info("\n--- YOLOv8n (Specialized) ---")
    try:
        spec = YOLO('best_cv_project.pt')
        spec.train(project='runs', name='specialized', **common)
        spec_best = 'runs/specialized/weights/best.pt'
        log.info(f"Specialized done: {spec_best}")
    except Exception as e:
        log.error(f"Specialized training failed: {e}", exc_info=True)
        raise

    log.info("\n--- YOLO11n (General) ---")
    try:
        gen = YOLO('best.pt')
        gen.train(project='runs', name='general', **common)
        gen_best = 'runs/general/weights/best.pt'
        log.info(f"General done: {gen_best}")
    except Exception as e:
        log.error(f"General training failed: {e}", exc_info=True)
        raise

    return spec_best, gen_best

def evaluate(model_path, name, val_dir):
    model = YOLO(model_path); model.to(DEVICE)
    imgs = sorted(Path(val_dir).glob('*.jpg'))[:200]
    dets_all, gts_all = [], []

    for imp in imgs:
        img = cv2.imread(str(imp)); h, w = img.shape[:2]
        lp = Path(val_dir.replace('images', 'labels')) / f"{imp.stem}.txt"
        gts = []
        if lp.exists():
            for line in open(lp):
                p = line.strip().split()
                if len(p) >= 5:
                    xc, yc, bw, bh = map(float, p[1:5])
                    gts.append([int((xc-bw/2)*w), int((yc-bh/2)*h),
                               int((xc+bw/2)*w), int((yc+bh/2)*h)])
        r = model(img, conf=0.15, iou=0.3, verbose=False)[0]
        dets = []
        if r.boxes:
            for b in r.boxes:
                dets.append({'conf': float(b.conf[0]),
                            'box': [float(x) for x in b.xyxy[0]]})
        dets_all.append(dets); gts_all.append(gts)

    # AP
    flat = []
    for i, ds in enumerate(dets_all):
        for d in ds: flat.append({'img': i, 'conf': d['conf'], 'box': d['box']})
    flat.sort(key=lambda x: x['conf'], reverse=True)

    gt_m = [set() for _ in gts_all]
    tp = np.zeros(len(flat)); fp = np.zeros(len(flat))
    total_gt = sum(len(g) for g in gts_all)

    for di, d in enumerate(flat):
        best_i, best_g = 0, -1
        for gi, g in enumerate(gts_all[d['img']]):
            if gi in gt_m[d['img']]: continue
            x1 = max(d['box'][0], g[0]); y1 = max(d['box'][1], g[1])
            x2 = min(d['box'][2], g[2]); y2 = min(d['box'][3], g[3])
            inter = max(0, x2-x1) * max(0, y2-y1)
            a1 = (d['box'][2]-d['box'][0])*(d['box'][3]-d['box'][1])
            a2 = (g[2]-g[0])*(g[3]-g[1])
            iou_v = inter / (a1 + a2 - inter + 1e-10)
            if iou_v > best_i: best_i, best_g = iou_v, gi
        if best_g >= 0 and best_i >= 0.5:
            tp[di] = 1; gt_m[d['img']].add(best_g)
        else:
            fp[di] = 1

    tp_c = np.cumsum(tp); fp_c = np.cumsum(fp)
    rec = tp_c / total_gt if total_gt > 0 else tp_c
    prec = tp_c / (tp_c + fp_c + 1e-10)
    ap = sum(np.max(prec[rec >= t]) if np.any(rec >= t) else 0
             for t in np.linspace(0, 1, 11)) / 11

    print(f"{name}: AP@0.5={ap:.4f}, total_gt={total_gt}")
    return {'ap': round(ap, 4), 'total_gt': total_gt}

def run():
    log.info("Starting experiment at %s", datetime.now().isoformat())
    try:
        _run()
        log.info("Experiment completed successfully at %s", datetime.now().isoformat())
    except Exception as e:
        log.error("Experiment failed: %s", e, exc_info=True)
        raise

def _run():
    # Step 1: Train
    spec_best, gen_best = train_models()

    # Step 2: Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)
    val_dir = f'{DATA}/images/val'
    spec_r = evaluate(spec_best, 'YOLOv8n Specialized', val_dir)
    gen_r = evaluate(gen_best, 'YOLO11n General', val_dir)

    # Step 3: Hungarian vs Greedy
    print("\n--- Hungarian vs Greedy ---")
    mask = cv2.imread('parking_mask.png', 0)
    cc = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    _, _, vals, _ = cc

    def iou(b1, b2):
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        a1 = (b1[2]-b1[0])*(b1[3]-b1[1]); a2 = (b2[2]-b2[0])*(b2[3]-b2[1])
        return inter / (a1 + a2 - inter + 1e-10)

    ROIS = [(int(vals[i,0]), int(vals[i,1]),
             int(vals[i,0]+vals[i,2]), int(vals[i,1]+vals[i,3]))
            for i in range(1, cc[0]) if vals[i,4] > 500]

    model = YOLO(spec_best); model.to(DEVICE)
    h_w, g_w, tie = 0, 0, 0
    for imp in sorted(Path(val_dir).glob('*.jpg'))[:50]:
        img = cv2.imread(str(imp))
        r = model(img, conf=0.15, verbose=False)[0]
        dets = [(int(b[0]), int(b[1]), int(b[2]), int(b[3]))
                for b in r.boxes.xyxy] if r.boxes else []

        # Greedy
        pairs = [(iou(ROIS[ri], d), ri, di)
                 for ri, ro in enumerate(ROIS) for di, d in enumerate(dets)
                 if iou(ROIS[ri], d) > 0.15]
        pairs.sort(reverse=True)
        g_match, used = {}, set()
        for iv, ri, di in pairs:
            if ri not in g_match and di not in used:
                g_match[ri] = di; used.add(di)

        # Hungarian
        nr, nd = len(ROIS), len(dets)
        if nd > 0:
            cost = np.ones((nr, nd))
            for i_, ro in enumerate(ROIS):
                for j, d in enumerate(dets):
                    iv = iou(ro, d)
                    cost[i_, j] = 1.0 - iv if iv > 0.15 else 1.0
            ra, da = linear_sum_assignment(cost)
            h_match = {int(ra[k]): int(da[k]) for k in range(len(ra))
                       if cost[ra[k], da[k]] < 0.85}
        else:
            h_match = {}

        gi = sum(iou(ROIS[ri], dets[di]) for ri, di in g_match.items())
        hi = sum(iou(ROIS[ri], dets[di]) for ri, di in h_match.items())
        if hi > gi + 1e-8: h_w += 1
        elif gi > hi + 1e-8: g_w += 1
        else: tie += 1

    print(f"Hungarian wins: {h_w}, Greedy wins: {g_w}, Ties: {tie}")

    # Step 4: Save
    results = {
        'timestamp': datetime.now().isoformat(),
        'device': 'RTX 5060 Ti 8GB',
        'dataset': 'PKLot (12,142 images)',
        'specialized': spec_r,
        'general': gen_r,
        'hungarian_vs_greedy': {
            'hungarian_wins': h_w, 'greedy_wins': g_w, 'ties': tie
        },
    }
    with open(f'{OUT}/results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {OUT}/results.json")
    print(json.dumps(results, indent=2))

if __name__ == '__main__':
    freeze_support()
    run()
