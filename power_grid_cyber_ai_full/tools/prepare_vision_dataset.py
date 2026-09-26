"""
Dataset Preparation Tool for Power Grid Vision Security

Downloads and prepares public datasets for training YOLO models:
- Fire detection datasets
- Person/PPE detection
- Intruder/anomaly detection

Usage:
    python tools/prepare_vision_dataset.py --output data/vision_datasets/
"""

import os
import argparse
from pathlib import Path


def create_sample_dataset(output_dir: str):
    """
    Create a sample dataset structure for training.
    In production, replace with actual dataset download (Kaggle, Roboflow, etc.)
    
    Dataset structure (YOLO format):
    datasets/
        fire_detection/
            images/
                train/
                val/
            labels/
                train/
                val/
            classes.txt
    """
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    
    # Dataset configurations
    datasets = {
        'fire_detection': {
            'classes': ['fire', 'smoke'],
            'description': 'Fire and smoke detection for power grid monitoring'
        },
        'intruder_detection': {
            'classes': ['person', 'vehicle', 'no_ppe'],
            'description': 'Intruder and PPE compliance detection'
        },
        'equipment_status': {
            'classes': ['breaker_open', 'breaker_closed', 'damaged_equipment'],
            'description': 'Equipment status classification'
        }
    }
    
    for ds_name, config in datasets.items():
        ds_dir = base / ds_name
        
        # Create YOLO structure
        for split in ['train', 'val']:
            (ds_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (ds_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
        
        # Write classes.txt
        classes_file = ds_dir / 'classes.txt'
        with open(classes_file, 'w') as f:
            for cls in config['classes']:
                f.write(f"{cls}\n")
        
        # Write dataset.yaml for YOLO training
        yaml_content = f"""# {config['description']}
path: {ds_dir.absolute()}
train: images/train
val: images/val

nc: {len(config['classes'])}
names: {config['classes']}
"""
        (ds_dir / 'dataset.yaml').write_text(yaml_content)
        
        # Create placeholder README
        readme = ds_dir / 'README.md'
        readme.write_text(f"""# {ds_name}

{config['description']}

## Classes
{chr(10).join(f"- {i}: {c}" for i, c in enumerate(config['classes']))}

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
""")
        
        print(f"✓ Created dataset structure: {ds_dir}")
    
    # Create master README
    master_readme = base / 'README.md'
    master_readme.write_text("""# Vision Datasets for Power Grid Security

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
""")
    
    print(f"\n✓ Dataset preparation complete: {base}")
    print("\nNext steps:")
    print("1. Populate datasets with real images")
    print("2. Run: yolo train data=<dataset>/dataset.yaml model=yolov8n.pt")


def download_from_kaggle(dataset_name: str, output_dir: str):
    """
    Download dataset from Kaggle (requires kaggle.json credentials).
    
    Example datasets:
    - vipure/fire-detection
    - andrewmvd/person-in-wild
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        output_path = Path(output_dir) / dataset_name.split('/')[1]
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading {dataset_name}...")
        api.dataset_download_files(dataset_name, path=str(output_path), unzip=True)
        print(f"✓ Downloaded to {output_path}")
        
    except ImportError:
        print("⚠ kaggle package not installed. Install with: pip install kaggle")
    except Exception as e:
        print(f"⚠ Download failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare vision datasets for power grid security training'
    )
    parser.add_argument(
        '--output', '-o',
        default='data/vision_datasets',
        help='Output directory for datasets'
    )
    parser.add_argument(
        '--kaggle', '-k',
        action='append',
        metavar='DATASET',
        help='Download dataset from Kaggle (e.g., vipure/fire-detection)'
    )
    
    args = parser.parse_args()
    
    # Create sample structure
    create_sample_dataset(args.output)
    
    # Download from Kaggle if specified
    if args.kaggle:
        for dataset in args.kaggle:
            download_from_kaggle(dataset, args.output)


if __name__ == '__main__':
    main()
