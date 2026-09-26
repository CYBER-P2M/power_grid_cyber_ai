# Vision Datasets for Power Grid Security

This directory contains datasets for training computer vision models.

## Quick Start

1. Populate datasets with real images (see each subfolder's README)
2. Train YOLO model:
   ```bash
   yolo train data=fire_detection/dataset.yaml model=yolov8n.pt epochs=100
   ```
3. Model will be saved to `runs/detect/train/weights/best.pt`

## Dataset Sources

### Fire/Smoke Detection
- Kaggle: "Fire and Smoke Dataset"
- Mendeley Data: "Fire Detection Images"

### Person/PPE Detection
- Roboflow: "PPE Detection"
- Kaggle: "Hardhat Detection"

### Equipment Status
- Custom annotation from substation cameras
- IEEE dataset imagery (if available)

## Annotation Format (YOLO)

Each label file (.txt) contains:
```
class_id x_center y_center width height
```

All values normalized to [0, 1].

## Next Steps

1. Collect minimum 500 images per class
2. Split 80% train, 20% val
3. Train and evaluate mAP@0.5
