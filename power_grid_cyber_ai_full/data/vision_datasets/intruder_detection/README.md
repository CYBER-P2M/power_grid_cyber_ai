# intruder_detection

Intruder and PPE compliance detection

## Classes
- 0: person
- 1: vehicle
- 2: no_ppe

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
