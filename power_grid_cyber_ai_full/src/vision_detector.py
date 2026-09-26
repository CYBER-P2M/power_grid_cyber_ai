"""
Vision Detector Module for Multimodal Cyber-Physical Power Grid Security

This module provides computer vision capabilities to detect physical anomalies
in power grid infrastructure (smoke, fire, intruders, equipment status).

Usage:
    detector = VisionDetector()
    result = detector.detect_anomalies(image_path)
    print(result['risk_score'], result['labels'])
"""

import os
from typing import Dict, List, Any, Optional
import numpy as np


class VisionDetector:
    """
    Computer vision detector for power grid physical security.
    
    Supports:
    - Smoke/fire detection
    - Intruder detection (person without PPE)
    - Equipment status (open/closed breaker, damaged equipment)
    - General anomaly scoring
    """
    
    # Predefined threat categories
    THREAT_CATEGORIES = {
        'fire_smoke': {'weight': 0.95, 'physical_risk': True},
        'intruder': {'weight': 0.85, 'physical_risk': True},
        'equipment_damage': {'weight': 0.80, 'physical_risk': True},
        'no_ppe': {'weight': 0.60, 'physical_risk': False},
        'normal': {'weight': 0.05, 'physical_risk': False}
    }
    
    def __init__(self, model_path: Optional[str] = None, threshold: float = 0.5):
        """
        Initialize the vision detector.
        
        Args:
            model_path: Path to pre-trained model (YOLO, etc.). If None, uses demo mode.
            threshold: Detection confidence threshold (0-1)
        """
        self.model_path = model_path
        self.threshold = threshold
        self.model = None
        
        # In production, load your YOLO/Custom model here
        # self.model = YOLO(model_path) if model_path else None
        
    def detect_anomalies(self, image_input: Any) -> Dict[str, Any]:
        """
        Detect physical anomalies in an image.
        
        Args:
            image_input: Can be:
                - str: path to image file
                - np.ndarray: image array (H, W, C)
                - dict: with 'frame' key containing image data
        
        Returns:
            dict with keys:
                - risk_score: float 0-1
                - labels: list of detected threat labels
                - confidence: dict of label -> confidence
                - is_physical_threat: bool
                - timestamp: detection timestamp
        """
        import time
        
        # Handle different input types
        if isinstance(image_input, dict):
            image = image_input.get('frame', image_input.get('image', None))
        elif isinstance(image_input, str):
            # In real implementation: image = cv2.imread(image_input)
            image = image_input  # placeholder
        else:
            image = image_input
        
        # Demo mode: simulate detection based on filename or random
        if self.model is None:
            return self._demo_detect(image)
        
        # Production mode: run model inference
        return self._model_detect(image)
    
    def _demo_detect(self, image_input: Any) -> Dict[str, Any]:
        """
        Demo detection mode for testing without a real model.
        Simulates detections based on image path or random sampling.
        """
        import time
        
        # Extract filename if path provided
        if isinstance(image_input, str):
            fname = os.path.basename(image_input).lower()
            
            # Simulate based on filename keywords
            if 'fire' in fname or 'smoke' in fname:
                return self._make_result(['fire_smoke'], [0.92])
            elif 'intruder' in fname or 'person' in fname:
                return self._make_result(['intruder', 'no_ppe'], [0.88, 0.75])
            elif 'damage' in fname or 'broken' in fname:
                return self._make_result(['equipment_damage'], [0.85])
        
        # Default: normal operation (low risk)
        return self._make_result(['normal'], [0.95])
    
    def _model_detect(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Run actual model inference.
        Replace this with your YOLO/Custom model logic.
        """
        # Example with YOLO:
        # results = self.model(image)
        # detections = results[0].boxes
        # 
        # labels = []
        # confidences = []
        # for box in detections:
        #     cls_id = int(box.cls[0])
        #     conf = float(box.conf[0])
        #     if conf > self.threshold:
        #         labels.append(self.class_names[cls_id])
        #         confidences.append(conf)
        
        # Placeholder
        return self._make_result(['normal'], [0.95])
    
    def _make_result(self, labels: List[str], confidences: List[float]) -> Dict[str, Any]:
        """
        Format detection results with risk scoring.
        """
        import time
        
        # Calculate risk score as weighted max
        risk_score = 0.0
        is_physical_threat = False
        confidence = {}
        
        for label, conf in zip(labels, confidences):
            if label in self.THREAT_CATEGORIES:
                cat = self.THREAT_CATEGORIES[label]
                weighted_risk = cat['weight'] * conf
                risk_score = max(risk_score, weighted_risk)
                if cat['physical_risk']:
                    is_physical_threat = True
            confidence[label] = conf
        
        return {
            'risk_score': float(risk_score),
            'labels': labels,
            'confidence': confidence,
            'is_physical_threat': is_physical_threat,
            'timestamp': time.time()
        }
    
    def process_video_stream(self, video_source: Any, callback=None):
        """
        Process a video stream frame-by-frame.
        
        Args:
            video_source: Camera index (int) or video file path (str)
            callback: Function to call for each frame result
        """
        import cv2
        
        cap = cv2.VideoCapture(video_source if isinstance(video_source, int) else str(video_source))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            result = self.detect_anomalies({'frame': frame})
            
            if callback:
                callback(result, frame)
        
        cap.release()


# Convenience function for quick testing
def quick_detect(image_path: str) -> Dict[str, Any]:
    """Quick detection on a single image."""
    detector = VisionDetector()
    return detector.detect_anomalies(image_path)


if __name__ == '__main__':
    # Test demo mode
    detector = VisionDetector()
    
    # Test with different simulated scenarios
    test_cases = [
        'camera_fire_smoke.jpg',
        'camera_intruder.jpg',
        'camera_normal.jpg'
    ]
    
    for test in test_cases:
        result = detector.detect_anomalies(test)
        print(f"{test}: risk={result['risk_score']:.2f}, labels={result['labels']}")
