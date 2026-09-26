# Multimodal Cyber-Physical Power Grid Security MVP

## Overview

This MVP adds **Computer Vision** capabilities to the existing cyber IDS system, enabling **multimodal fusion** for comprehensive security assessment.

## Architecture

```
┌─────────────────┐     ┌─────────────────┐
│   Cyber IDS     │     │  Vision Detector│
│  (SCADA/PMU)    │     │  (Cameras/YOLO) │
└────────┬────────┘     └────────┬────────┘
         │                       │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │   Multimodal Fusion   │
         │  (Late Fusion + CD)   │
         └───────────┬───────────┘
                     │
         ┌───────────▼───────────┐
         │   Dashboard (Streamlit)│
         │   4_Multimodal_CPS.py  │
         └───────────────────────┘
```

## New Files

| File | Purpose |
|------|--------|
| `src/vision_detector.py` | Computer vision anomaly detection (demo + production mode) |
| `app/multimodal_fusion.py` | Late fusion engine with contradiction detection |
| `dashboard/pages/4_Multimodal_CPS.py` | Streamlit dashboard page for multimodal view |
| `data/multimodal_demo.csv` | Demo dataset with fused decisions |
| `MULTIMODAL_MVP.md` | This documentation |

## Decision Categories

| Category | Cyber | Physical | Meaning |
|----------|-------|----------|---------|
| CYBER_PHYSICAL_EVENT | High | High | Coordinated attack or cascade failure |
| CYBER_ATTACK | High | Low | Remote intrusion, FDIA, DoS |
| PHYSICAL_FAULT | Low | High | Equipment failure, fire, damage |
| INTRUSION | Low | Medium | Unauthorized access, no PPE |
| NORMAL | Low | Low | All systems normal |

## Fusion Formula

\[
R_{fused} = w_c \cdot R_{cyber} + w_p \cdot R_{physical}
\]

\[
R_{adjusted} = R_{fused} \times M_{category}
\]

Where:
- \(w_c + w_p = 1\) (default: 0.55 / 0.45)
- \(M_{category}\) is the category multiplier (CYBER_PHYSICAL_EVENT = 1.2, etc.)

## Usage

### 1. Run Dashboard

```bash
cd power_grid_cyber_ai_full
cstreamlit run dashboard/pages/4_Multimodal_CPS.py
```

### 2. Test Fusion Programmatically

```python
from app.multimodal_fusion import MultimodalFusion
from src.vision_detector import VisionDetector

fusion = MultimodalFusion()
vision = VisionDetector()

# Simulate inputs
cyber = {'risk_score': 0.85, 'attack_type': 'FDIA', 'confidence': 0.9}
vision_result = vision.detect_anomalies('camera_fire.jpg')

# Fuse
fused = fusion.fuse(cyber, vision_result)
print(fused['decision_category'], fused['fused_risk_score'])
```

### 3. Integrate Real Models

Replace demo mode in `vision_detector.py`:

```python
from ultralytics import YOLO

class VisionDetector:
    def __init__(self, model_path='yolov8.pt'):
        self.model = YOLO(model_path)
    
    def detect_anomalies(self, image):
        results = self.model(image)
        # Process detections...
```

## Next Steps

1. **Train custom YOLO model** on power grid imagery (fire, intruders, PPE, equipment)
2. **Connect real camera feeds** (RTSP streams from substations)
3. **Add temporal fusion** (LSTM/Transformer over time-series of fused decisions)
4. **Deploy with Docker** alongside existing dashboard

## References

- IEEE 118-Bus Power Grid Dataset (cyber-physical scenarios)
- YOLOv8 for real-time object detection
- Late fusion strategies for multimodal security

---

**Author:** Multimodal CPS Security Team  
**Date:** September 2026
