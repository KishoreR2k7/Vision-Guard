"""
Complete Training Plan - Garbage & Violence Detection
Step-by-step guide to train all remaining models
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
from ultralytics import YOLO
import torch
import yaml

def check_datasets():
    """Check which datasets are available and ready"""
    print("=" * 70)
    print("📊 DATASET STATUS CHECK")
    print("=" * 70)
    
    datasets = {
        'accident': {
            'path': Path('dataset/accident_detetcion/data1'),
            'status': '✅ TRAINED',
            'ready': False
        },
        'garbage': {
            'path': Path('dataset/garbage/data1'),
            'status': 'Pending',
            'ready': False
        },
        'violence': {
            'path': Path('dataset/violence'),
            'status': 'Empty',
            'ready': False
        }
    }
    
    # Check garbage dataset
    garbage_path = datasets['garbage']['path']
    if garbage_path.exists():
        train_imgs = list((garbage_path / 'train' / 'images').glob('*')) if (garbage_path / 'train' / 'images').exists() else []
        if len(train_imgs) > 0:
            datasets['garbage']['status'] = f'✅ Ready ({len(train_imgs)} training images)'
            datasets['garbage']['ready'] = True
        else:
            datasets['garbage']['status'] = '❌ No images found'
    else:
        datasets['garbage']['status'] = '❌ Path not found'
    
    # Check violence dataset
    violence_path = datasets['violence']['path']
    if violence_path.exists():
        violence_imgs = list((violence_path / 'images').glob('*')) if (violence_path / 'images').exists() else []
        if len(violence_imgs) > 0:
            datasets['violence']['status'] = f'✅ Ready ({len(violence_imgs)} images)'
            datasets['violence']['ready'] = True
        else:
            datasets['violence']['status'] = '❌ Empty - Need to add images'
    else:
        datasets['violence']['status'] = '❌ Path not found'
    
    # Print status
    print("\n📁 Dataset Status:")
    for name, info in datasets.items():
        print(f"   {name.upper():15} - {info['status']}")
    
    return datasets

def fix_numpy_issue():
    """Check and suggest numpy fix"""
    print("\n" + "=" * 70)
    print("🔧 NUMPY COMPATIBILITY CHECK")
    print("=" * 70)
    
    try:
        import numpy as np
        print(f"\n✅ NumPy version: {np.__version__}")
        
        # Check if numpy._core exists
        try:
            import numpy._core
            print("✅ numpy._core module available")
            return True
        except ImportError:
            print("⚠️ numpy._core not available (may cause issues)")
            return False
    except Exception as e:
        print(f"❌ NumPy error: {e}")
        return False

def train_garbage():
    """Train garbage detection model"""
    print("\n" + "=" * 70)
    print("🗑️  TRAINING: GARBAGE DETECTION")
    print("=" * 70)
    
    dataset_path = Path("dataset/garbage/data1")
    yaml_path = dataset_path / "data.yaml"
    
    # Update data.yaml
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
    
    print(f"\n📂 Dataset: {dataset_path}")
    print(f"📄 Config: {yaml_path}")
    
    # Train
    print("\n🚀 Starting training...\n")
    
    model = YOLO("yolo11n.pt")
    
    results = model.train(
        data=str(yaml_path),
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
    
    print("\n✅ Garbage detection training complete!")
    print("📁 Model saved: runs/train/garbage_detection/weights/best.pt")
    
    return True

def main():
    print("\n" + "=" * 70)
    print("🎯 VISION GUARD - COMPLETE TRAINING PLAN")
    print("=" * 70)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("\n⚠️ No GPU - Using CPU")
    
    # Check datasets
    datasets = check_datasets()
    
    # Check numpy
    numpy_ok = fix_numpy_issue()
    
    # Show action plan
    print("\n" + "=" * 70)
    print("📋 ACTION PLAN")
    print("=" * 70)
    
    print("\n✅ COMPLETED:")
    print("   1. Accident Detection - Already trained (99.5% mAP)")
    print("      Model: runs/train/accident_detection/weights/best.pt")
    
    print("\n⏭️  NEXT STEPS:")
    
    if datasets['garbage']['ready']:
        print("\n   2. Garbage Detection - Ready to train")
        if not numpy_ok:
            print("      ⚠️ Fix numpy first: pip install --upgrade numpy")
        else:
            print("      ▶️ Run this script to train!")
    else:
        print("\n   2. Garbage Detection - ❌ Dataset not ready")
    
    if datasets['violence']['ready']:
        print("\n   3. Violence Detection - Ready to train")
    else:
        print("\n   3. Violence Detection - ❌ Dataset empty")
        print("      Options:")
        print("      a) Add your own violence videos/images to dataset/violence/")
        print("      b) Use existing dataset_prepared/ (if available)")
        print("      c) Download violence detection dataset")
    
    # Offer to train garbage if ready and numpy OK
    if datasets['garbage']['ready'] and numpy_ok:
        print("\n" + "=" * 70)
        choice = input("\n🚀 Train garbage detection now? (y/n): ").strip().lower()
        
        if choice == 'y':
            try:
                train_garbage()
            except Exception as e:
                print(f"\n❌ Training failed: {e}")
                print("\n💡 Try fixing numpy:")
                print("   pip install numpy==1.26.4")
                print("   or")
                print("   conda install numpy=1.26 -c conda-forge")
    else:
        print("\n" + "=" * 70)
        print("\n💡 RECOMMENDATIONS:")
        
        if not numpy_ok:
            print("\n1. Fix NumPy compatibility:")
            print("   pip install numpy==1.26.4")
            print("   or")
            print("   conda install numpy=1.26 -c conda-forge")
        
        if not datasets['garbage']['ready']:
            print("\n2. Check garbage dataset:")
            print("   - Verify images exist in dataset/garbage/data1/train/images/")
            print("   - Verify labels exist in dataset/garbage/data1/train/labels/")
        
        if not datasets['violence']['ready']:
            print("\n3. For violence detection:")
            print("   Option A: Use prepared dataset")
            print("   - Check if dataset_prepared/ has images")
            print("   ")
            print("   Option B: Download violence dataset")
            print("   - Download from Kaggle/Roboflow")
            print("   - Place in dataset/violence/")
            print("   ")
            print("   Option C: Use custom videos")
            print("   - Place violence videos in dataset/violence/videos/")
            print("   - Run frame extraction script")

if __name__ == "__main__":
    main()
