# save as: tools/download_fire_dataset.py
import os
import zipfile
import urllib.request

output_dir = "data/vision_datasets/fire_detection/images/train"
os.makedirs(output_dir, exist_ok=True)

# مثال: Dataset صغير من Roboflow
url = "https://universe.roboflow.com/dataset/fire-detection/images/100.zip"
zip_path = "temp_fire.zip"

print(f"Downloading from {url}...")
urllib.request.urlretrieve(url, zip_path)

print("Extracting...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(output_dir)

os.remove(zip_path)
print(f"✓ Done! Images in {output_dir}")