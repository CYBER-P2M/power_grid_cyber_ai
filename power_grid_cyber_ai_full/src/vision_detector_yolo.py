"""
Production Vision Detector with YOLOv8

Replaces demo mode with actual YOLO inference for real-time detection.

Requirements:
    pip install ultralytics opencv-python

Usage:
    detector = VisionDetectorYOLO(model_path='models/fire_detector.pt')
    result = detector.detect_anomalies('camera_feed.jpg')
"""

import os
from typing import Dict, List, Any, Optional, Union
import numpy as np


class VisionDetectorYOLO:
    """
    Production-grade vision detector using YOLOv8.
    
    Supports:
    - Single image inference
    - Video stream processing
    - RTSP camera feeds
    - Multi-model ensemble (fire + intruder + equipment)
    """
    
    # Class mappings for different models
    CLASS_MAPPINGS = {
        'fire_model': {
            0: 'fire',
            1: 'smoke'
        },
        'intruder_model': {
            0: 'person',
            1: 'vehicle',
            2: 'no_ppe'
        },
        'equipment_model': {
            0: 'breaker_open',
            1: 'breaker_closed',
            2: 'damaged_equipment'
        }
    }
    
    # Risk scores per class
    CLASS_RISK = {
        'fire': 0.95,
        'smoke': 0.90,
        'person': 0.70,
        'vehicle': 0.50,
        'no_ppe': 0.60,
        'intruder': 0.85,
        'breaker_open': 0.40,
        'breaker_closed': 0.05,
        'damaged_equipment': 0.80
    }
    
    def __init__(
        self,
        model_paths: Optional[Union[str, List[str]]] = None,
        confidence_threshold: float = 0.5,
        iou_threshold: float = 0.45,
        device: str = 'cpu'
    ):
        """
        Initialize YOLO detector.
        
        Args:
            model_paths: Path(s) to trained .pt model file(s). Can be:
                - str: single model path
                - list: multiple models for ensemble
                - None: use default YOLOv8n (demo)
            confidence_threshold: Detection confidence threshold (0-1)
            iou_threshold: NMS IoU threshold (0-1)
            device: 'cpu', 'cuda', 'cuda:0', etc.
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.models = []
        self.model_names = []
        
        # Import ultralytics
        try:
            from ultralytics import YOLO
            self.YOLO = YOLO
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics"
            )
        
        # Load models
        if model_paths:
            if isinstance(model_paths, str):
                model_paths = [model_paths]
            
            for path in model_paths:
                if os.path.exists(path):
                    model = self.YOLO(path)
                    model.to(device)
                    self.models.append(model)
                    self.model_names.append(os.path.basename(path))
                    print(f"✓ Loaded model: {path}")
                else:
                    print(f"⚠ Model not found: {path}")
        
        # Fallback to default YOLOv8n if no models loaded
        if not self.models:
            print("⚠ No custom models loaded, using YOLOv8n (demo mode)")
            self.models = [self.YOLO('yolov8n.pt')]
            self.model_names = ['yolov8n.pt']
    
    def detect_anomalies(self, image_input: Any) -> Dict[str, Any]:
        """
        Detect anomalies in an image using loaded YOLO models.
        
        Args:
            image_input: Can be:
                - str: path to image file
                - np.ndarray: image array (H, W, C) in BGR or RGB
                - dict: with 'frame' or 'image' key
        
        Returns:
            dict with keys:
                - risk_score: float 0-1
                - labels: list of detected labels
                - confidence: dict of label -> confidence
                - is_physical_threat: bool
                - detections: list of raw detection dicts
                - timestamp: float
        """
        import time
        import cv2
        
        # Handle input types
        if isinstance(image_input, dict):
            image = image_input.get('frame', image_input.get('image', None))
        elif isinstance(image_input, str):
            image = cv2.imread(image_input)
            if image is None:
                return self._error_result(f"Failed to load image: {image_input}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, np.ndarray):
            image = image_input
            if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 3 and image.dtype != np.uint8:
                # Assume RGB float [0,1]
                image = (image * 255).astype(np.uint8)
        else:
            return self._error_result(f"Unsupported input type: {type(image_input)}")
        
        # Run inference with all models
        all_detections = []
        
        for model, model_name in zip(self.models, self.model_names):
            results = model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Get class name
                    class_name = result.names[cls_id]
                    
                    # Get bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    all_detections.append({
                        'label': class_name,
                        'confidence': conf,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'model': model_name
                    })
        
        # Calculate risk score
        risk_score = 0.0
        labels = []
        confidence = {}
        is_physical_threat = False
        
        for det in all_detections:
            label = det['label']
            conf = det['confidence']
            
            # Get risk for this class
            class_risk = self.CLASS_RISK.get(label, 0.5)
            weighted_risk = class_risk * conf
            
            risk_score = max(risk_score, weighted_risk)
            
            if label not in labels:
                labels.append(label)
            
            # Aggregate confidence for same label
            if label in confidence:
                confidence[label] = max(confidence[label], conf)
            else:
                confidence[label] = conf
            
            # Check if physical threat
            if label in ['fire', 'smoke', 'person', 'intruder', 'damaged_equipment']:
                is_physical_threat = True
        
        return {
            'risk_score': float(risk_score),
            'labels': labels,
            'confidence': confidence,
            'is_physical_threat': is_physical_threat,
            'detections': all_detections,
            'timestamp': time.time()
        }
    
    def _error_result(self, message: str) -> Dict[str, Any]:
        """Return error result."""
        import time
        return {
            'risk_score': 0.0,
            'labels': ['error'],
            'confidence': {'error': 1.0},
            'is_physical_threat': False,
            'detections': [],
            'error': message,
            'timestamp': time.time()
        }
    
    def process_camera(
        self,
        camera_source: Union[int, str],
        callback=None,
        display: bool = False
    ):
        """
        Process live camera feed.
        
        Args:
            camera_source: Camera index (int) or RTSP URL (str)
            callback: Function(result, frame) called for each frame
            display: Show OpenCV window (for local testing)
        """
        import cv2
        
        cap = cv2.VideoCapture(
            camera_source if isinstance(camera_source, int) else str(camera_source)
        )
        
        if not cap.isOpened():
            print(f"⚠ Failed to open camera: {camera_source}")
            return
        
        print(f"✓ Camera opened: {camera_source}")
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠ Frame capture failed")
                break
            
            # Convert BGR to RGB for YOLO
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect
            result = self.detect_anomalies({'frame': frame_rgb})
            
            # Draw bounding boxes
            if display:
                for det in result.get('detections', []):
                    x1, y1, x2, y2 = map(int, det['bbox'])
                    label = det['label']
                    conf = det['confidence']
                    
                    # Color based on risk
                    if label in ['fire', 'smoke']:
                        color = (0, 0, 255)  # Red
                    elif label == 'person':
                        color = (0, 255, 255)  # Cyan
                    else:
                        color = (0, 255, 0)  # Green
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        frame,
                        f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )
                
                # Show risk score
                risk = result['risk_score']
                cv2.putText(
                    frame,
                    f"Risk: {risk:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255) if risk > 0.6 else (0, 255, 0),
                    2
                )
                
                cv2.imshow('Power Grid Security', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Callback
            if callback:
                callback(result, frame)
            
            frame_count += 1
            
            # Log every 100 frames
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames, risk={result['risk_score']:.2f}")
        
        cap.release()
        
        if display:
            cv2.destroyAllWindows()
        
        print(f"✓ Camera processing complete: {frame_count} frames")


# Convenience function
def quick_detect_yolo(image_path: str, model_path: Optional[str] = None) -> Dict[str, Any]:
    """Quick detection with YOLO."""
    detector = VisionDetectorYOLO(model_paths=model_path)
    return detector.detect_anomalies(image_path)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', '-i', type=str, help='Test image path')
    parser.add_argument('--camera', '-c', type=int, default=0, help='Camera index')
    parser.add_argument('--model', '-m', type=str, help='Model .pt path')
    parser.add_argument('--display', '-d', action='store_true', help='Show display')
    
    args = parser.parse_args()
    
    detector = VisionDetectorYOLO(model_paths=args.model)
    
    if args.image:
        result = detector.detect_anomalies(args.image)
        print(f"Risk: {result['risk_score']:.2f}")
        print(f"Labels: {result['labels']}")
        print(f"Confidence: {result['confidence']}")
    else:
        print(f"Testing camera {args.camera}...")
        detector.process_camera(args.camera, display=args.display)
