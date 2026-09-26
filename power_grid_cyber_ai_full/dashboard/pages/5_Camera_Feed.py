"""
Streamlit Dashboard Page: Live Camera Feed with YOLO Detection.

Displays real-time video feed with computer-vision anomaly detection.

Requirements:
    pip install opencv-python streamlit pandas numpy

Usage:
    streamlit run dashboard/pages/5_Camera_Feed.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from datetime import datetime
from typing import Any

import cv2
import numpy as np
import pandas as pd
import streamlit as st


# Add project root and src directory to Python path
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


from src.vision_detector_yolo import VisionDetectorYOLO
from app.multimodal_fusion import MultimodalFusion


# -------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Live Camera Feed",
    page_icon="📹",
    layout="wide",
)

st.title("📹 Live Camera Feed - Vision Security")
st.markdown(
    """
    **مباشر من الكاميرا**: كشف الحريق، المتسللين، وحالة المعدات في الوقت الفعلي.  
    يستخدم YOLOv8 للكشف مع دمج الإشارات للقرار الأمني.
    """
)


# -------------------------------------------------------------------
# Session state initialization
# -------------------------------------------------------------------
if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

if "detection_log" not in st.session_state:
    st.session_state.detection_log = []

if "camera_frame_count" not in st.session_state:
    st.session_state.camera_frame_count = 0


# -------------------------------------------------------------------
# Sidebar configuration
# -------------------------------------------------------------------
st.sidebar.header("⚙️ Camera Settings")

camera_source = st.sidebar.text_input(
    "Camera Source",
    value="0",
    help="Camera index such as 0, 1, 2, or an RTSP URL.",
)

model_path = st.sidebar.text_input(
    "Model Path",
    value="",
    placeholder="models/fire_detector.pt",
    help="Path to a trained YOLO model (.pt). Leave empty for demo/default model.",
)

confidence = st.sidebar.slider(
    "Confidence Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.05,
)


# -------------------------------------------------------------------
# Detector initialization
# -------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLO model...")
def load_detector(selected_model_path: str, conf: float) -> VisionDetectorYOLO:
    if selected_model_path and os.path.exists(selected_model_path):
        return VisionDetectorYOLO(
            model_paths=[selected_model_path],
            confidence_threshold=conf,
        )

    return VisionDetectorYOLO(confidence_threshold=conf)


try:
    detector = load_detector(model_path, confidence)
except Exception as error:
    st.error(f"Failed to load YOLO detector: {error}")
    st.stop()


# Fusion engine is retained for integration with the multimodal page.
fusion = MultimodalFusion()


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------
def get_result_value(
    result: dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    """Safely read an item from detector output."""
    return result.get(key, default)


def normalize_labels(labels: Any) -> list[str]:
    """Return labels as a clean list of strings."""
    if labels is None:
        return []

    if isinstance(labels, str):
        return [labels]

    if isinstance(labels, (list, tuple, set)):
        return [str(label) for label in labels if str(label).strip()]

    return []


def extract_bbox(detection: dict[str, Any]) -> tuple[int, int, int, int] | None:
    """Extract a bounding box from common YOLO result formats."""
    bbox = detection.get("bbox", detection.get("box"))

    if bbox is None:
        return None

    if isinstance(bbox, dict):
        try:
            return (
                int(bbox.get("x1", 0)),
                int(bbox.get("y1", 0)),
                int(bbox.get("x2", 0)),
                int(bbox.get("y2", 0)),
            )
        except (TypeError, ValueError):
            return None

    if isinstance(bbox, (list, tuple, np.ndarray)) and len(bbox) >= 4:
        try:
            x1, y1, x2, y2 = bbox[:4]
            return int(x1), int(y1), int(x2), int(y2)
        except (TypeError, ValueError):
            return None

    return None


def detection_color(label: str) -> tuple[int, int, int]:
    """
    Return OpenCV BGR colors.
    Red = fire/smoke, Cyan = person, Green = other.
    """
    normalized_label = label.lower().strip()

    if normalized_label in {"fire", "smoke", "flame"}:
        return (0, 0, 255)  # Red in BGR

    if normalized_label in {"person", "intruder", "human"}:
        return (255, 255, 0)  # Cyan in BGR

    return (0, 255, 0)  # Green in BGR


def draw_detections(
    frame: np.ndarray,
    detections: list[dict[str, Any]],
) -> np.ndarray:
    """Draw YOLO bounding boxes and labels onto the OpenCV frame."""
    height, width = frame.shape[:2]

    for detection in detections:
        bbox = extract_bbox(detection)

        if bbox is None:
            continue

        x1, y1, x2, y2 = bbox

        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(0, min(x2, width - 1))
        y2 = max(0, min(y2, height - 1))

        label = str(
            detection.get(
                "label",
                detection.get("class_name", "object"),
            )
        )

        try:
            detection_confidence = float(
                detection.get(
                    "confidence",
                    detection.get("conf", 0.0),
                )
            )
        except (TypeError, ValueError):
            detection_confidence = 0.0

        color = detection_color(label)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label_y = max(25, y1 - 10)
        cv2.putText(
            frame,
            f"{label} {detection_confidence:.2f}",
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return frame


def add_detection_log(
    risk: float,
    labels: list[str],
    is_threat: bool,
) -> None:
    """Append a detection record and retain only the latest 100 records."""
    if not labels or labels == ["error"]:
        return

    st.session_state.detection_log.append(
        {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "risk": round(float(risk), 2),
            "labels": ", ".join(labels),
            "is_threat": bool(is_threat),
        }
    )

    st.session_state.detection_log = st.session_state.detection_log[-100:]


# -------------------------------------------------------------------
# Dashboard layout
# -------------------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("🎥 Live Feed")

    frame_placeholder = st.empty()
    status_text = st.empty()

    button_col1, button_col2 = st.columns(2)

    start_button = button_col1.button(
        "▶️ Start",
        type="primary",
        use_container_width=True,
    )

    stop_button = button_col2.button(
        "⏹️ Stop",
        use_container_width=True,
    )

    if start_button:
        st.session_state.camera_running = True

    if stop_button:
        st.session_state.camera_running = False
        status_text.info("Camera stopped.")

with col2:
    st.subheader("📊 Detection Results")

    risk_metric = st.empty()
    labels_metric = st.empty()

    risk_metric.metric("Risk Score", "--")
    labels_metric.metric("Labels", "--")

    st.subheader("📜 Recent Detections")

    log_container = st.empty()

    if st.session_state.detection_log:
        initial_log_df = pd.DataFrame(
            st.session_state.detection_log[-10:]
        )

        log_container.dataframe(
            initial_log_df,
            hide_index=True,
            use_container_width=True,
        )
    else:
        log_container.info("No detections recorded yet.")

    action_col1, action_col2 = st.columns(2)

    export_log = action_col1.button(
        "📥 Export Log",
        use_container_width=True,
    )

    clear_log = action_col2.button(
        "🗑️ Clear Log",
        use_container_width=True,
    )

    if clear_log:
        st.session_state.detection_log = []
        log_container.info("Detection log cleared.")

    if export_log:
        if st.session_state.detection_log:
            export_df = pd.DataFrame(st.session_state.detection_log)
            csv_data = export_df.to_csv(index=False).encode("utf-8")

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            st.download_button(
                label="⬇️ Download CSV",
                data=csv_data,
                file_name=f"detection_log_{timestamp}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.warning("There are no detections to export.")


# -------------------------------------------------------------------
# Camera processing
# -------------------------------------------------------------------
def process_frame() -> None:
    """Open camera, detect objects, update video feed and metrics."""
    try:
        parsed_camera_source: int | str = int(camera_source)
    except ValueError:
        parsed_camera_source = camera_source.strip()

    cap = cv2.VideoCapture(parsed_camera_source)

    if not cap.isOpened():
        status_text.error(
            f"Failed to open camera source: {camera_source}. "
            "Check the camera index, permissions, or RTSP URL."
        )
        return

    status_text.success(f"Camera connected: {camera_source}")

    frame_count = 0

    try:
        while st.session_state.camera_running and cap.isOpened():
            success, frame = cap.read()

            if not success or frame is None:
                status_text.warning("Frame capture failed. Camera feed stopped.")
                break

            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                result = detector.detect_anomalies(
                    {"frame": frame_rgb}
                )

                if not isinstance(result, dict):
                    result = {}

            except Exception as error:
                status_text.error(f"Detection error: {error}")
                break

            try:
                risk = float(get_result_value(result, "risk_score", 0.0))
            except (TypeError, ValueError):
                risk = 0.0

            risk = max(0.0, min(1.0, risk))

            labels = normalize_labels(
                get_result_value(result, "labels", [])
            )

            detections = get_result_value(result, "detections", [])
            if not isinstance(detections, list):
                detections = []

            is_physical_threat = bool(
                get_result_value(result, "is_physical_threat", False)
            )

            frame = draw_detections(frame, detections)

            overlay_color = (
                (0, 0, 255) if risk > 0.60 else (0, 255, 0)
            )

            cv2.putText(
                frame,
                f"Risk: {risk:.2f}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                overlay_color,
                2,
                cv2.LINE_AA,
            )

            frame_placeholder.image(
                frame,
                channels="BGR",
                use_container_width=True,
            )

            risk_metric.metric("Risk Score", f"{risk:.2f}")

            labels_metric.metric(
                "Labels",
                ", ".join(labels) if labels else "--",
            )

            add_detection_log(
                risk=risk,
                labels=labels,
                is_threat=is_physical_threat,
            )

            if st.session_state.detection_log:
                recent_log_df = pd.DataFrame(
                    st.session_state.detection_log[-10:]
                )

                log_container.dataframe(
                    recent_log_df,
                    hide_index=True,
                    use_container_width=True,
                )

            frame_count += 1
            st.session_state.camera_frame_count = frame_count

            if frame_count % 10 == 0:
                status_text.info(f"Processing: {frame_count} frames")

            # Small delay reduces CPU usage while maintaining a live feed.
            time.sleep(0.03)

    finally:
        cap.release()

        if not st.session_state.camera_running:
            status_text.info("Camera stopped and released.")
        else:
            status_text.warning("Camera released.")


# -------------------------------------------------------------------
# Run camera when Start is pressed
# -------------------------------------------------------------------
if st.session_state.camera_running:
    with st.spinner("Starting camera..."):
        process_frame()


# -------------------------------------------------------------------
# Test uploaded image
# -------------------------------------------------------------------
st.divider()
st.subheader("🖼️ Test with Image")

uploaded = st.file_uploader(
    "Upload image for testing",
    type=["jpg", "png", "jpeg"],
    key="camera_test_upload",
)

if uploaded is not None:
    file_suffix = os.path.splitext(uploaded.name)[1].lower()

    if file_suffix not in {".jpg", ".jpeg", ".png"}:
        file_suffix = ".jpg"

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=file_suffix,
    ) as temp_file:
        temp_file.write(uploaded.getvalue())
        temp_path = temp_file.name

    try:
        image_result = detector.detect_anomalies(temp_path)

        if not isinstance(image_result, dict):
            image_result = {}

        image_risk = float(
            get_result_value(image_result, "risk_score", 0.0)
        )

        image_labels = normalize_labels(
            get_result_value(image_result, "labels", [])
        )

        image_detections = get_result_value(
            image_result,
            "detections",
            [],
        )

        image_threat = bool(
            get_result_value(
                image_result,
                "is_physical_threat",
                False,
            )
        )

        image_confidence = get_result_value(
            image_result,
            "confidence",
            0.0,
        )

        image_col, result_col = st.columns(2)

        with image_col:
            st.image(
                uploaded,
                caption="Uploaded Image",
                use_container_width=True,
            )

        with result_col:
            st.json(
                {
                    "risk_score": image_risk,
                    "labels": image_labels,
                    "is_physical_threat": image_threat,
                    "confidence": image_confidence,
                    "detections": (
                        image_detections[:5]
                        if isinstance(image_detections, list)
                        else []
                    ),
                }
            )

    except Exception as error:
        st.error(f"Could not process uploaded image: {error}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
st.divider()

active_model = (
    model_path
    if model_path and os.path.exists(model_path)
    else "YOLO default/demo model"
)

st.caption(
    "Live Camera Feed | YOLOv8 Detection | "
    f"Model: {active_model} | "
    f"Confidence: {confidence:.0%}"
)