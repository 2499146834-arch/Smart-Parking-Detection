#!/usr/bin/env python3
"""
Smart Parking Detection System - Improved Version
- CV Project View: Specialized model + 396 predefined ROIs (high precision)
- General View: YOLO11n direct detection (generalization)
- Improvements: Hungarian matching, YAML config, feature-based view detection
"""

import os
import sys
import cv2
import yaml
import base64
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'qtUI'))
from ultralytics import YOLO

app = Flask(__name__)

# ===================== Load Configuration =====================
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

cfg = load_config()

app.config['MAX_CONTENT_LENGTH'] = cfg.get('server', {}).get('max_upload_mb', 500) * 1024 * 1024
CORS(app)

# ===================== Configuration =====================
BASE_DIR = os.path.dirname(__file__)
GENERAL_MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), 'models', 'general_best.pt')
SPECIALIZED_MODEL_PATH = os.path.join(os.path.dirname(BASE_DIR), 'models', 'specialized_best.pt')
MASK_PATH = os.path.join(BASE_DIR, cfg.get('models', {}).get('mask', 'parking_mask.png'))

# Thresholds from config
th = cfg.get('thresholds', {})
CV_CONF_THRESHOLD = th.get('cv_confidence', 0.15)
GENERAL_CONF_THRESHOLD = th.get('general_confidence', 0.3)
VIEW_DETECTION_THRESHOLD = th.get('view_detection_count', 25)
IOU_THRESHOLD = th.get('iou', 0.3)
ROI_MATCH_THRESHOLD = th.get('roi_match_iou', 0.15)

# View detection parameters
view_cfg = th.get('view_detection', {})
VIEW_WINDOW_SIZE = view_cfg.get('multi_frame_window', 5)
VIEW_MIN_VOTES = view_cfg.get('multi_frame_min_votes', 4)

# Video parameters
vid_cfg = cfg.get('video', {})
MAX_FRAMES = vid_cfg.get('max_frames', 150)
FRAME_SKIP = vid_cfg.get('frame_skip', 3)
CV_RESOLUTION = vid_cfg.get('cv_project_resolution', [1920, 1080])

print(f"[Config] Loaded: CV conf={CV_CONF_THRESHOLD}, General conf={GENERAL_CONF_THRESHOLD}, "
      f"View threshold={VIEW_DETECTION_THRESHOLD}, ROI IoU={ROI_MATCH_THRESHOLD}")

# 全局模型
general_model = None
specialized_model = None

# Predefined ROI
PREDEFINED_ROIS = []

# ===================== 初始化 =====================
def load_predefined_rois():
    global PREDEFINED_ROIS
    if os.path.exists(MASK_PATH):
        mask = cv2.imread(MASK_PATH, 0)
        if mask is not None:
            connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
            (totalLabels, label_ids, values, centroid) = connected_components
            
            rois = []
            min_area = 500
            for i in range(1, totalLabels):
                area = values[i, cv2.CC_STAT_AREA]
                if area > min_area:
                    x1 = int(values[i, cv2.CC_STAT_LEFT])
                    y1 = int(values[i, cv2.CC_STAT_TOP])
                    w = int(values[i, cv2.CC_STAT_WIDTH])
                    h = int(values[i, cv2.CC_STAT_HEIGHT])
                    rois.append((x1, y1, x1 + w, y1 + h))
            
            PREDEFINED_ROIS = rois
            print(f"✓ Predefined ROI: {len(rois)} parking spaces")
            return True
    return False

load_predefined_rois()

def get_general_model():
    global general_model
    if general_model is None:
        print("Loading general model (YOLO11n)...")
        general_model = YOLO(GENERAL_MODEL_PATH)
        print("✓ 通用Model loaded")
    return general_model

def get_specialized_model():
    global specialized_model
    if specialized_model is None:
        print("Loading specialized model (YOLOv8n)...")
        specialized_model = YOLO(SPECIALIZED_MODEL_PATH)
        print("✓ 专用Model loaded")
    return specialized_model

# ===================== View Detection =====================

# Reference template for feature-based view verification (lazy init)
_reference_descriptors = None
_orb = None

def _init_orb():
    global _orb
    if _orb is None:
        _orb = cv2.ORB_create(nfeatures=500)

def _load_reference_template():
    """Load and cache ORB descriptors from the parking mask for fast scene matching."""
    global _reference_descriptors
    if _reference_descriptors is not None:
        return _reference_descriptors
    mask = cv2.imread(MASK_PATH, 0)
    if mask is not None:
        _init_orb()
        kp = _orb.detect(mask, None)
        kp, des = _orb.compute(mask, kp)
        _reference_descriptors = des
    return _reference_descriptors

def detect_view_type(img):
    """Detect view type: count-based primary + ORB feature backup verification."""
    model = get_specialized_model()
    results = model(img, verbose=False, conf=CV_CONF_THRESHOLD)[0]
    detection_count = len(results.boxes)

    print(f"  View detection: {detection_count} objects (threshold: {VIEW_DETECTION_THRESHOLD})")

    # Primary: object count threshold
    if detection_count < VIEW_DETECTION_THRESHOLD:
        return 'general', detection_count

    # Secondary verification: ORB feature matching against reference mask
    ref_des = _load_reference_template()
    if ref_des is not None:
        _init_orb()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = _orb.detectAndCompute(gray, None)
        if des is not None and len(des) > 10:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(ref_des, des)
            avg_distance = np.mean([m.distance for m in matches]) if matches else float('inf')
            print(f"  ORB verification: {len(matches)} matches, avg distance={avg_distance:.1f}")
            # If feature distance is too large, scene doesn't match known view
            if avg_distance > 60:
                print("  ORB rejected: scene does not match CV Project reference")
                return 'general', detection_count

    return 'specialized', detection_count

def calculate_iou(box1, box2):
    """计算两个框的IOU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0


def roi_based_detect(img, rois, model, conf_thresh):
    """基于ROI的检测（专用模式）- 匈牙利算法最优匹配"""
    results = model(img, conf=conf_thresh, iou=IOU_THRESHOLD, verbose=False)[0]

    # 提取所有检测框
    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = results.names[cls_id]
        detections.append({
            'bbox': (x1, y1, x2, y2),
            'conf': conf,
            'class': cls_name,
            'class_id': cls_id,
        })

    n_rois = len(rois)
    n_dets = len(detections)

    if n_dets == 0:
        return [{'id': i + 1, 'bbox': [rx1, ry1, rx2, ry2],
                 'class': 'available', 'confidence': 0.0}
                for i, (rx1, ry1, rx2, ry2) in enumerate(rois)]

    # Build cost matrix: cost = 1 - IoU (Hungarian minimizes cost)
    cost_matrix = np.ones((n_rois, n_dets), dtype=np.float32)
    for i, (rx1, ry1, rx2, ry2) in enumerate(rois):
        roi_box = (rx1, ry1, rx2, ry2)
        for j, det in enumerate(detections):
            iou = calculate_iou(roi_box, det['bbox'])
            cost_matrix[i, j] = 1.0 - iou if iou > ROI_MATCH_THRESHOLD else 1.0

    # Hungarian algorithm for optimal global assignment
    roi_indices, det_indices = linear_sum_assignment(cost_matrix)

    # Filter valid matches (cost < 1 - threshold means IoU > threshold)
    valid_matches = {}
    for r_idx, d_idx in zip(roi_indices, det_indices):
        if cost_matrix[r_idx, d_idx] < (1.0 - ROI_MATCH_THRESHOLD):
            valid_matches[r_idx] = d_idx

    # Build results
    roi_results = []
    for i, (rx1, ry1, rx2, ry2) in enumerate(rois):
        if i in valid_matches:
            det = detections[valid_matches[i]]
            cls_name = det['class'].lower()
            if cls_name in ['car', 'occupied', 'vehicle', 'full']:
                status = 'full'
            else:
                status = 'available'
            confidence = det['conf']
        else:
            status = 'available'
            confidence = 0.0

        roi_results.append({
            'id': i + 1,
            'bbox': [rx1, ry1, rx2, ry2],
            'class': status,
            'confidence': round(confidence, 3)
        })

    print(f"  Hungarian matching: {len(valid_matches)}/{n_dets} detections matched to {n_rois} ROIs")

    return roi_results

def direct_yolo_detect(img, model, conf_thresh):
    """YOLO直接检测（通用模式）"""
    results = model(img, conf=conf_thresh, iou=IOU_THRESHOLD, verbose=False)[0]
    
    detections = []
    for i, box in enumerate(results.boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = results.names[cls_id]
        
        # 类别映射
        if cls_name.lower() in ['full', 'occupied', 'car', 'vehicle', 'illegal']:
            display_class = 'full'
        else:
            display_class = 'available'
        
        detections.append({
            'id': i + 1,
            'bbox': [x1, y1, x2, y2],
            'class': display_class,
            'confidence': round(conf, 3)
        })
    
    return detections

def draw_detections(img, detections, title=""):
    """绘制检测Result"""
    result = img.copy()
    h, w = result.shape[:2]
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        color = (0, 0, 255) if det['class'] == 'full' else (0, 255, 0)
        
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        label = f"{det['class']} {det['confidence']:.2f}" if det['confidence'] > 0 else det['class']
        label_y = y1 - 5 if y1 > 20 else y2 + 15
        
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(result, (x1, label_y - text_h - 2), (x1 + text_w, label_y + 2), color, -1)
        cv2.putText(result, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    occupied = sum(1 for d in detections if d['class'] == 'full')
    empty = sum(1 for d in detections if d['class'] == 'available')
    
    cv2.rectangle(result, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.putText(result, f"{title} | Total: {len(detections)} | Occupied: {occupied} | Available: {empty}", 
                (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return result

# ===================== API =====================
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok',
        'model_loaded': True,
        'predefined_roi_count': len(PREDEFINED_ROIS),
        'mode': 'final'
    })

@app.route('/api/classes', methods=['GET'])
def get_classes():
    return jsonify({
        'classes': ['full', 'available'],
        'names': {'full': 'Occupied', 'available': 'Available'}
    })

@app.route('/api/detect/image', methods=['POST'])
def detect_image():
    start_time = datetime.now()
    
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Empty file'}), 400
        
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'error': 'Cannot decode image'}), 400
        
        print(f"\nProcessing image: {file.filename}, size: {img.shape}")
        
        # 检测视角
        view_type, detection_count = detect_view_type(img)
        
        if view_type == 'specialized' and PREDEFINED_ROIS:
            # CV Project View：使用专用模型 + ROI
            print("  → Using CV Project specialized mode")
            model = get_specialized_model()
            detections = roi_based_detect(img, PREDEFINED_ROIS, model, CV_CONF_THRESHOLD)
            model_name = 'CV Project YOLOv8n (ROI精准检测)'
        else:
            # General View：使用YOLO11n直接检测
            print("  → Using general YOLO detection mode")
            model = get_general_model()
            detections = direct_yolo_detect(img, model, GENERAL_CONF_THRESHOLD)
            model_name = 'YOLO11n (通用检测)'
        
        # 绘制Result
        result_img = draw_detections(img, detections, model_name)
        
        # 转换为base64
        _, buffer = cv2.imencode('.png', result_img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # 统计
        occupied = sum(1 for d in detections if d['class'] == 'full')
        empty = sum(1 for d in detections if d['class'] == 'available')
        
        inference_time = (datetime.now() - start_time).total_seconds()
        
        # 格式化detections以匹配前端
        formatted_detections = []
        for det in detections:
            formatted_detections.append({
                'id': det['id'],
                'class': det['class'],
                'confidence': det['confidence'],
                'bbox': {
                    'xmin': det['bbox'][0],
                    'ymin': det['bbox'][1],
                    'xmax': det['bbox'][2],
                    'ymax': det['bbox'][3]
                }
            })
        
        print(f"  Result: {len(detections)} objects, occupied:{occupied}, available:{empty}, time:{inference_time:.3f}s")
        
        return jsonify({
            'detections': formatted_detections,
            'count': len(detections),
            'occupied': occupied,
            'empty': empty,
            'image': img_base64,
            'model_used': model_name,
            'view_type': view_type,
            'inference_time': round(inference_time, 3),
            'filename': file.filename
        })
        
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/detect/video', methods=['POST'])
def detect_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'Empty file'}), 400
        
        upload_path = '/tmp/upload_video.mp4'
        file.save(upload_path)
        
        cap = cv2.VideoCapture(upload_path)
        if not cap.isOpened():
            return jsonify({'error': 'Cannot open video'}), 400
        
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 15
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"  Video: {width}x{height}, Total frames: {total_frames}")
        
        is_cv_project = False
        frame_checks = []
        
        # Check multiple frames for robust view validation
        check_frames = [0, min(100, total_frames // 4), min(200, total_frames // 2)]
        for frame_idx in check_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, check_frame = cap.read()
            if ret:
                _, count = detect_view_type(check_frame)
                frame_checks.append(count)
                print(f"    帧{frame_idx}: {count} objects")
        
        # 判断是否为CV Project View
        # 条件：视频size匹配 + 多帧平均检测数≥threshold
        cv_w, cv_h = CV_RESOLUTION
        if width == cv_w and height == cv_h and len(frame_checks) >= 2:
            avg_detections = sum(frame_checks) / len(frame_checks)
            if avg_detections >= VIEW_DETECTION_THRESHOLD - 2:  # 放宽到23
                is_cv_project = True
                print(f"  ✓ 确认为CV Project View (平均: {avg_detections:.1f})")
            else:
                print(f"  ✗ 非CV Project View (平均: {avg_detections:.1f} < 23)")
        else:
            print(f"  ✗ size不匹配或非CV Project视频")
        
        if is_cv_project:
            model = get_specialized_model()
            conf_thresh = CV_CONF_THRESHOLD
            rois = PREDEFINED_ROIS
            model_name = 'CV Project ROI'
        else:
            model = get_general_model()
            conf_thresh = GENERAL_CONF_THRESHOLD
            rois = None
            model_name = 'YOLO11n'
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        output_path = '/tmp/output_video.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        max_frames = min(total_frames, MAX_FRAMES)
        
        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % FRAME_SKIP == 0:
                if rois:
                    detections = roi_based_detect(frame, rois, model, conf_thresh)
                else:
                    detections = direct_yolo_detect(frame, model, conf_thresh)
                result_frame = draw_detections(frame, detections, model_name)
            else:
                result_frame = frame
            
            out.write(result_frame)
            frame_count += 1
        
        cap.release()
        out.release()
        
        return send_file(output_path, mimetype='video/mp4', as_attachment=False)
        
    except Exception as e:
        import traceback
        return jsonify({'error': str(e)}), 500

@app.route('/')
def serve_frontend():
    frontend_path = os.path.join(os.path.dirname(__file__), '..', 'frontend', 'index.html')
    if os.path.exists(frontend_path):
        return open(frontend_path, encoding='utf-8').read()
    return '<h1>Frontend not found</h1>', 404

if __name__ == '__main__':
    print(f"\n{'='*50}")
    print("Smart Parking Detection Server")
    print(f"Frontend: http://localhost:8000")
    print(f"API Docs: http://localhost:8000/api/health")
    print(f"{'='*50}\n")
    app.run(host='0.0.0.0', port=8000, debug=True)
