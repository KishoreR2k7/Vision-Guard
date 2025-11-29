"""
Quick Training Script - Train YOLO models on your new datasets
This script automatically prepares and trains models for all available datasets
"""

import os
import sys
from pathlib import Path

# Fix OpenMP library conflict
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ultralytics import YOLO
import torch
import yaml

def setup_accident_dataset():
    """Setup accident detection dataset"""
    dataset_path = Path("dataset/accident_detetcion/data1")
    yaml_path = dataset_path / "data.yaml"
    
    if not dataset_path.exists():
        print(f"❌ Accident dataset not found")
        return None
    
    # Create data.yaml
    data = {
        'path': str(dataset_path.absolute()),
        'train': 'train/Accident',
        'val': 'val/Accident',
        'test': 'test/Accident',
        'nc': 1,
        'names': ['Accident']
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    # Create full-frame labels for all images
    print("   Creating labels for accident images...")
    for split in ['train', 'val', 'test']:
        img_dir = dataset_path / split / 'Accident'
        if img_dir.exists():
            count = 0
            for img_file in img_dir.glob('*.jpg'):
                label_file = img_file.with_suffix('.txt')
                if not label_file.exists():
                    with open(label_file, 'w') as f:
                        f.write("0 0.5 0.5 1.0 1.0\n")
                    count += 1
            for img_file in img_dir.glob('*.png'):
                label_file = img_file.with_suffix('.txt')
                if not label_file.exists():
                    with open(label_file, 'w') as f:
                        f.write("0 0.5 0.5 1.0 1.0\n")
                    count += 1
            print(f"   ✅ Created {count} labels in {split}")
    
    return yaml_path

def setup_garbage_dataset():
    """Setup garbage detection dataset"""
    dataset_path = Path("dataset/garbage/data1")
    yaml_path = dataset_path / "data.yaml"
    
    if not dataset_path.exists():
        print(f"❌ Garbage dataset not found")
        return None
    
    # Fix data.yaml paths
    data = {
        'path': str(dataset_path.absolute()),
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': 1,
        'names': ['garbage']
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    return yaml_path

def train_model(data_yaml, model_name, epochs=100):
    """Train a YOLO model"""
    
    print(f"\n{'='*70}")
    print(f"🚀 Training: {model_name}")
    print(f"{'='*70}")
    
    # Load model
    model = YOLO("yolo11n.pt")
    
    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=16,
        imgsz=640,
        device=0 if torch.cuda.is_available() else 'cpu',
        project='runs/train',
        name=model_name,
        patience=20,
        save=True,
        pretrained=True,
        optimizer='AdamW',
        lr0=0.01,
        amp=True,
        verbose=True,
        workers=2,  # Reduced workers to avoid conflicts
    )
    
    print(f"\n✅ Training complete: {model_name}")
    print(f"📁 Best model: runs/train/{model_name}/weights/best.pt")
    
    return results

def main():
    print("=" * 70)
    print("🎯 VISION GUARD AI - QUICK TRAINING")
    print("=" * 70)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("\n⚠️ Training on CPU (slower)")
    
    # Setup datasets
    print("\n📚 Setting up datasets...")
    
    datasets = {}
    
    print("\n1. Accident Detection:")
    accident_yaml = setup_accident_dataset()
    if accident_yaml:
        datasets['accident_detection'] = accident_yaml
        print("   ✅ Ready")
    
    print("\n2. Garbage Detection:")
    garbage_yaml = setup_garbage_dataset()
    if garbage_yaml:
        datasets['garbage_detection'] = garbage_yaml
        print("   ✅ Ready")
    
    if not datasets:
        print("\n❌ No datasets found!")
        return
    
    print(f"\n{'='*70}")
    print(f"Found {len(datasets)} dataset(s) ready for training")
    print(f"{'='*70}")
    
    # Train each model
    for model_name, data_yaml in datasets.items():
        try:
            train_model(data_yaml, model_name, epochs=100)
        except Exception as e:
            print(f"\n❌ Error training {model_name}: {e}")
    
    print("\n" + "=" * 70)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 70)
    print("\n📊 Check results in:")
    for model_name in datasets.keys():
        print(f"   • runs/train/{model_name}/")
    
    print("\n💡 Use the best.pt models for inference!")

if __name__ == "__main__":
    main()
