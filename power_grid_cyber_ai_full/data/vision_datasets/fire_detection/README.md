# fire_detection

Fire and smoke detection for power grid monitoring

## Classes
- 0: fire
- 1: smoke

## Data Collection

To populate this dataset:
1. Collect images from power grid cameras
2. Annotate with YOLO format (class_id, x_center, y_center, width, height)
3. Place in images/train, images/val
4. Corresponding labels in labels/train, labels/val

## Recommended Sources
- Kaggle: Fire detection datasets
- Roboflow Universe: PPE detection
- Custom: Substation camera footage
