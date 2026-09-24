"""
Streamlit Dashboard Page: Multimodal Cyber-Physical Security

Displays fused risk assessment combining cyber IDS and computer vision.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import sys

# Add parent paths for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'app')))

from app.multimodal_fusion import MultimodalFusion
from src.vision_detector import VisionDetector


# Page config
st.set_page_config(
    page_title="Multimodal CPS Security",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ Multimodal Cyber-Physical Power Grid Security")
st.markdown("""
**دمج الإشارات السيبرانية والفيزيائية** لقرار أمني شامل.
يجمع هذا النظام بين كشف التسلل الشبكي (IDS) والرؤية الحاسوبية (Computer Vision)
لتشخيص الهجمات المنسقة والهجمات السيبرانية-الفيزيائية.
""")

# Initialize fusion engine
fusion = MultimodalFusion(cyber_weight=0.55, physical_weight=0.45)
vision = VisionDetector()

# Sidebar: Controls
st.sidebar.header("⚙️ Settings")
cyber_weight = st.sidebar.slider("Cyber Weight", 0.0, 1.0, 0.55, 0.05)
physical_weight = st.sidebar.slider("Physical Weight", 0.0, 1.0, 0.45, 0.05)

if abs(cyber_weight + physical_weight - 1.0) > 0.01:
    st.sidebar.error("Weights must sum to 1.0")
    st.stop()

fusion = MultimodalFusion(cyber_weight=cyber_weight, physical_weight=physical_weight)

# Demo mode toggle
demo_mode = st.sidebar.checkbox("🎭 Demo Mode", value=True)

# Main layout: 3 columns
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.subheader("🔌 Cyber IDS Input")
    
    if demo_mode:
        cyber_risk = st.slider("Cyber Risk Score", 0.0, 1.0, 0.3, 0.05)
        attack_type = st.selectbox(
            "Attack Type",
            ['normal', 'FDIA', 'DoS', 'replay', 'data_injection']
        )
        cyber_conf = st.slider("Cyber Confidence", 0.0, 1.0, 0.85, 0.05)
    else:
        # Placeholder for real integration
        st.info("Connect to real IDS model output")
        cyber_risk = 0.3
        attack_type = 'normal'
        cyber_conf = 0.85
    
    cyber_result = {
        'risk_score': cyber_risk,
        'attack_type': attack_type,
        'confidence': cyber_conf
    }
    
    st.metric("Cyber Risk", f"{cyber_risk:.2f}")
    st.caption(f"Type: {attack_type}")

with col2:
    st.subheader("📷 Vision Input")
    
    if demo_mode:
        physical_risk = st.slider("Physical Risk Score", 0.0, 1.0, 0.15, 0.05)
        vision_labels = st.multiselect(
            "Detected Labels",
            ['normal', 'fire_smoke', 'intruder', 'no_ppe', 'equipment_damage'],
            default=['normal']
        )
        is_threat = 'fire_smoke' in vision_labels or 'intruder' in vision_labels or 'equipment_damage' in vision_labels
    else:
        st.info("Connect to camera feed / YOLO model")
        physical_risk = 0.15
        vision_labels = ['normal']
        is_threat = False
    
    vision_result = {
        'risk_score': physical_risk,
        'labels': vision_labels,
        'is_physical_threat': is_threat,
        'confidence': {label: 0.9 for label in vision_labels}
    }
    
    st.metric("Physical Risk", f"{physical_risk:.2f}")
    st.caption(f"Labels: {', '.join(vision_labels)}")

with col3:
    st.subheader("🔀 Fused Decision")
    
    # Perform fusion
    fused = fusion.fuse(cyber_result, vision_result)
    
    # Color-coded risk display
    risk_level = fused['fused_risk_score']
    if risk_level > 0.8:
        risk_color = "🔴 CRITICAL"
    elif risk_level > 0.6:
        risk_color = "🟠 HIGH"
    elif risk_level > 0.4:
        risk_color = "🟡 MEDIUM"
    else:
        risk_color = "🟢 LOW"
    
    st.metric("Fused Risk", f"{risk_level:.2f} {risk_color}")
    st.metric("Category", fused['decision_category'])
    
    # Contribution breakdown
    st.caption("**Contribution:**")
    st.progress(fused['cyber_contribution'])
    st.caption(f"Cyber: {fused['cyber_contribution']:.2f}")
    st.progress(fused['physical_contribution'])
    st.caption(f"Physical: {fused['physical_contribution']:.2f}")
    
    if fused['is_contradiction']:
        st.warning("⚠️ Signal contradiction detected - verify sensors")

# Full explanation
st.divider()
st.subheader("📋 Diagnostic Explanation")
st.info(fused['explanation'])

# Decision table
st.divider()
st.subheader("📊 Decision Matrix")

decision_df = pd.DataFrame({
    'Cyber': [cyber_risk],
    'Physical': [physical_risk],
    'Fused': [fused['fused_risk_score']],
    'Category': [fused['decision_category']],
    'Contradiction': [fused['is_contradiction']]
})

st.dataframe(decision_df, hide_index=True, use_container_width=True)

# Historical log (session state)
if 'fusion_history' not in st.session_state:
    st.session_state.fusion_history = []

if st.button("📝 Log Current Decision"):
    st.session_state.fusion_history.append({
        'timestamp': time.strftime('%H:%M:%S'),
        'cyber': cyber_risk,
        'physical': physical_risk,
        'fused': fused['fused_risk_score'],
        'category': fused['decision_category']
    })

if st.session_state.fusion_history:
    st.divider()
    st.subheader("📜 Recent Decisions")
    history_df = pd.DataFrame(st.session_state.fusion_history[-10:])
    st.dataframe(history_df, hide_index=True, use_container_width=True)

# Upload image for vision test
st.divider()
st.subheader("🖼️ Test Vision Detector")

uploaded = st.file_uploader("Upload test image", type=['jpg', 'png', 'jpeg'])

if uploaded:
    # Save temporarily
    temp_path = f"/tmp/{uploaded.name}"
    with open(temp_path, 'wb') as f:
        f.write(uploaded.getvalue())
    
    result = vision.detect_anomalies(temp_path)
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.image(uploaded, caption="Uploaded Image", use_container_width=True)
    with col_b:
        st.json({
            'risk_score': result['risk_score'],
            'labels': result['labels'],
            'is_physical_threat': result['is_physical_threat'],
            'confidence': result['confidence']
        })

# Footer
st.divider()
st.caption(
    "Multimodal Cyber-Physical Security MVP | "
    "Fusion: Late fusion with contradiction detection | "
    "Weights: Cyber {:.0%} / Physical {:.0%}".format(cyber_weight, physical_weight)
)
