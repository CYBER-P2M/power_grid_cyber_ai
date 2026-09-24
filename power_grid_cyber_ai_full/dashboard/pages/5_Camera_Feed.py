"""
Streamlit Dashboard Page: Live Camera Feed with YOLO Detection

Displays real-time video feed with computer vision anomaly detection.

Requirements:
    pip install opencv-python streamlit

Usage:
    streamlit run dashboard/pages/5_Camera_Feed.py
"""

import streamlit as st
import cv2
import numpy as np
import os
import sys
import tempfile
from datetime import datetime

# Add parent paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))

from src.vision_detector_yolo import VisionDetectorYOLO
from app.multimodal_fusion import MultimodalFusion


# Page config
st.set_page_config(
    page_title="Live Camera Feed",
    page_icon="📹",
    layout="wide"
)

st.title("📹 Live Camera Feed - Vision Security")
st.markdown("""
**مباشر من الكاميرا**: كشف الحريق، المتسللين، وحالة المعدات في الوقت الفعلي.
يستخدم YOLOv8 للكشف مع دمج الإشارات للقرار الأمني.
""")

# Sidebar: Configuration
st.sidebar.header("⚙️ Camera Settings")

camera_source = st.sidebar.text_input(
    "Camera Source",
    value="0",
    help="Camera index (0, 1, 2) or RTSP URL"
)

model_path = st.sidebar.text_input(
    "Model Path",
    value="",
    placeholder="models/fire_detector.pt",
    help="Path to trained YOLO model (.pt)"
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.0, 1.0, 0.5, 0.05
)

# Initialize detector
@st.cache_resource
def load_detector(model_path: str, conf: float):
    if model_path and os.path.exists(model_path):
        return VisionDetectorYOLO(model_paths=[model_path], confidence_threshold=conf)
    else:
        return VisionDetectorYOLO(confidence_threshold=conf)

detector = load_detector(model_path, confidence)

# Fusion engine
fusion = MultimodalFusion()

# Main layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎥 Live Feed")
    
    # Placeholder for video frame
    frame_placeholder = st.empty()
    
    # Status
    status_text = st.empty()
    
    # Start/Stop buttons
    col_start, col_stop = st.columns(2)
    
    start_button = col_start.button("▶️ Start")
    stop_button = col_stop.button("⏹️ Stop")
    
    # Session state for control
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False
    
    if start_button:
        st.session_state.camera_running = True
    
    if stop_button:
        st.session_state.camera_running = False
        status_text.text("Camera stopped")

with col2:
    st.subheader("📊 Detection Results")
    
    # Metrics
    risk_metric = st.metric("Risk Score", "--")
    labels_metric = st.metric("Labels", "--")
    
    # Detection log
    st.subheader("📜 Recent Detections")
    
    if 'detection_log' not in st.session_state:
        st.session_state.detection_log = []
    
    log_container = st.empty()
    
    if st.session_state.detection_log:
        log_df = st.session_state.detection_log[-10:]
        log_container.dataframe(
            log_df,
            hide_index=True,
            use_container_width=True
        )
    
    # Export button
    if st.button("📥 Export Log"):
        if st.session_state.detection_log:
            import pandas as pd
            df = pd.DataFrame(st.session_state.detection_log)
            csv = df.to_csv(index=False)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"detection_log_{timestamp}.csv",
                mime="text/csv"
            )

# Camera processing
def process_frame():
    """Process camera frames in a loop."""
    
    # Parse camera source
    try:
        camera_idx = int(camera_source)
    except ValueError:
        camera_idx = camera_source  # RTSP URL
    
    cap = cv2.VideoCapture(camera_idx if isinstance(camera_idx, int) else str(camera_idx))
    
    if not cap.isOpened():
        status_text.error(f"Failed to open camera: {camera_source}")
        return
    
    status_text.success(f"Camera connected: {camera_source}")
    
    frame_count = 0
    
    while st.session_state.camera_running and cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            status_text.warning("Frame capture failed")
            break
        
        # Convert BGR to RGB for YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect
        result = detector.detect_anomalies({'frame': frame_rgb})
        
        # Draw bounding boxes
        for det in result.get('detections', []):
            x1, y1, x2, y2 = map(int, det['bbox'])
            label = det['label']
            conf = det['confidence']
            
            # Color based on risk
            if label in ['fire', 'smoke']:
                color = (255, 0, 0)  # Red (BGR)
            elif label == 'person':
                color = (255, 255, 0)  # Cyan
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
        
        # Add risk overlay
        risk = result['risk_score']
        overlay_color = (0, 0, 255) if risk > 0.6 else (0, 255, 0)
        cv2.putText(
            frame,
            f"Risk: {risk:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            overlay_color,
            2
        )
        
        # Display in Streamlit
        frame_placeholder.image(frame, channels="BGR", use_container_width=True)
        
        # Update metrics
        risk_metric.metric("Risk Score", f"{risk:.2f}")
        labels_metric.metric("Labels", ", ".join(result['labels']) if result['labels'] else "--")
        
        # Log detection
        if result['labels'] and result['labels'] != ['error']:
            st.session_state.detection_log.append({
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'risk': f"{risk:.2f}",
                'labels': ', '.join(result['labels']),
                'is_threat': result['is_physical_threat']
            })
            
            # Keep last 100
            if len(st.session_state.detection_log) > 100:
                st.session_state.detection_log = st.session_state.detection_log[-100:]
            
            log_container.dataframe(
                pd.DataFrame(st.session_state.detection_log[-10:]),
                hide_index=True,
                use_container_width=True
            )
        
        frame_count += 1
        
        # Throttle to ~10 FPS for Streamlit
        if frame_count % 3 == 0:
            status_text.text(f"Processing: {frame_count} frames")
    
    cap.release()
    status_text.text("Camera released")

# Run camera if started
if st.session_state.camera_running:
    with st.spinner("Starting camera..."):
        process_frame()

# Test image upload
st.divider()
st.subheader("🖼️ Test with Image")

uploaded = st.file_uploader(
    "Upload image for testing",
    type=['jpg', 'png', 'jpeg'],
    key="camera_test_upload"
)

if uploaded:
    temp_path = os.path.join(tempfile.gettempdir(), uploaded.name)
    with open(temp_path, 'wb') as f:
        f.write(uploaded.getvalue())
    
    result = detector.detect_anomalies(temp_path)
    
    col_img, col_res = st.columns(2)
    
    with col_img:
        st.image(uploaded, caption="Uploaded Image", use_container_width=True)
    
    with col_res:
        st.json({
            'risk_score': result['risk_score'],
            'labels': result['labels'],
            'is_physical_threat': result['is_physical_threat'],
            'confidence': result['confidence'],
            'detections': result.get('detections', [])[:5]  # Limit display
        })

# Footer
st.divider()
st.caption(
    "Live Camera Feed | YOLOv8 Detection | "
    f"Model: {model_path if model_path else 'yolov8n (demo)'} | "
    f"Confidence: {confidence:.0%}"
)
