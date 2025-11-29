"""
Train Garbage Detection Model Only
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
from ultralytics import YOLO
import torch
import yaml

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

def main():
    print("=" * 70)
    print("🗑️  GARBAGE DETECTION TRAINING")
    print("=" * 70)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("\n⚠️ Training on CPU")
    
    # Setup dataset
    print("\n📚 Setting up garbage detection dataset...")
    data_yaml = setup_garbage_dataset()
    
    if not data_yaml:
        print("❌ Dataset setup failed!")
        return
    
    print("   ✅ Dataset ready")
    
    # Train
    print("\n🚀 Starting training...\n")
    
    try:
        model = YOLO("yolo11n.pt")
        
        results = model.train(
            data=str(data_yaml),
            epochs=100,
            batch=16,
            imgsz=640,
            device=0 if torch.cuda.is_available() else 'cpu',
            project='runs/train',
            name='garbage_detection',
            patience=20,
            save=True,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.01,
            amp=True,
            verbose=True,
            workers=2,
        )
        
        print("\n" + "=" * 70)
        print("✅ GARBAGE DETECTION TRAINING COMPLETE!")
        print("=" * 70)
        print("\n📁 Results: runs/train/garbage_detection/")
        print("🎯 Best model: runs/train/garbage_detection/weights/best.pt")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\n💡 Try fixing with: pip install --upgrade numpy")

if __name__ == "__main__":
    main()
