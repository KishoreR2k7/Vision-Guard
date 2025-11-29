"""
Multi-Class Training Script for Vision Guard AI
Trains YOLO models for: Accident Detection, Garbage Detection, and Violence Detection
"""

import os
from pathlib import Path
from ultralytics import YOLO
import torch
import yaml

# Dataset configurations
DATASETS = {
    'accident': {
        'path': Path('../dataset/accident_detetcion/data1'),
        'name': 'accident_detection',
        'epochs': 100,
        'batch_size': 16,
        'image_size': 640,
    },
    'garbage': {
        'path': Path('../dataset/garbage/data1'),
        'name': 'garbage_detection',
        'epochs': 100,
        'batch_size': 16,
        'image_size': 640,
    },
    'violence': {
        'path': Path('../dataset_prepared'),  # Will use the prepared dataset
        'name': 'violence_detection',
        'epochs': 100,
        'batch_size': 16,
        'image_size': 640,
    }
}

# Model configuration
MODEL_SIZE = "yolo11n.pt"  # YOLOv11 Nano
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
LEARNING_RATE = 0.01
PATIENCE = 20  # Early stopping patience

def check_dataset_structure(dataset_path):
    """Check if dataset has proper YOLO structure"""
    required_dirs = ['train', 'val', 'test']
    
    # Check if it's using images/labels structure
    if (dataset_path / 'images' / 'train').exists():
        print(f"   ✅ Found YOLO format with images/ and labels/ structure")
        return True
    
    # Check for direct train/val/test structure
    for dir_name in required_dirs:
        if not (dataset_path / dir_name).exists():
            print(f"   ⚠️ Warning: Missing '{dir_name}' directory")
            return False
    
    print(f"   ✅ Dataset structure validated")
    return True

def create_data_yaml(dataset_name, dataset_path):
    """Create or update data.yaml for the dataset"""
    yaml_path = dataset_path / 'data.yaml'
    
    if yaml_path.exists():
        print(f"   ✅ Using existing data.yaml")
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        return yaml_path
    
    print(f"   ⚠️ data.yaml not found, checking structure...")
    
    # For accident detection
    if dataset_name == 'accident':
        data = {
            'path': str(dataset_path.absolute()),
            'train': 'train/Accident' if (dataset_path / 'train' / 'Accident').exists() else 'train',
            'val': 'val/Accident' if (dataset_path / 'val' / 'Accident').exists() else 'val',
            'test': 'test/Accident' if (dataset_path / 'test' / 'Accident').exists() else 'test',
            'nc': 1,
            'names': ['Accident']
        }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f)
    
    print(f"   ✅ Created data.yaml")
    return yaml_path

def train_model(dataset_key, config):
    """Train YOLO model for a specific dataset"""
    
    print("\n" + "=" * 70)
    print(f"🎯 Training Model: {config['name'].upper()}")
    print("=" * 70)
    
    dataset_path = config['path']
    
    # Check if dataset exists
    if not dataset_path.exists():
        print(f"❌ Error: Dataset path not found: {dataset_path}")
        return False
    
    print(f"📂 Dataset Path: {dataset_path.absolute()}")
    
    # Check dataset structure
    if not check_dataset_structure(dataset_path):
        print(f"⚠️ Warning: Dataset structure may be incomplete")
    
    # Create/verify data.yaml
    data_yaml = create_data_yaml(dataset_key, dataset_path)
    
    # Display training configuration
    print(f"\n📊 Training Configuration:")
    print(f"   Model: {MODEL_SIZE}")
    print(f"   Epochs: {config['epochs']}")
    print(f"   Batch Size: {config['batch_size']}")
    print(f"   Image Size: {config['image_size']}")
    print(f"   Device: {'GPU - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Mixed Precision: {'Enabled' if torch.cuda.is_available() else 'Disabled'}")
    
    # Load model
    print(f"\n🔄 Loading {MODEL_SIZE}...")
    model = YOLO(MODEL_SIZE)
    
    # Train
    print("\n🚀 Starting training...\n")
    
    try:
        results = model.train(
            data=str(data_yaml),
            epochs=config['epochs'],
            batch=config['batch_size'],
            imgsz=config['image_size'],
            device=DEVICE,
            project='runs/train',
            name=config['name'],
            lr0=LEARNING_RATE,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=3.0,
            patience=PATIENCE,
            save=True,
            save_period=10,
            pretrained=True,
            optimizer='AdamW',
            verbose=True,
            amp=torch.cuda.is_available(),
            
            # Data augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.0,
            copy_paste=0.0,
            
            # Performance
            workers=4,
            close_mosaic=10,
        )
        
        print("\n" + "=" * 70)
        print(f"✅ Training completed for {config['name']}!")
        print(f"📁 Results saved in: runs/train/{config['name']}")
        print(f"🎯 Best model: runs/train/{config['name']}/weights/best.pt")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        return False

def main():
    """Main training function"""
    
    print("\n" + "=" * 70)
    print("🚀 VISION GUARD AI - MULTI-CLASS TRAINING")
    print("=" * 70)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\n✅ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("\n⚠️ No GPU available - Training will be slower on CPU")
    
    # List available datasets
    print("\n📚 Available Datasets:")
    available_datasets = {}
    for key, config in DATASETS.items():
        if config['path'].exists():
            print(f"   ✅ {config['name']}: {config['path']}")
            available_datasets[key] = config
        else:
            print(f"   ❌ {config['name']}: NOT FOUND ({config['path']})")
    
    if not available_datasets:
        print("\n❌ No datasets found! Please check your dataset paths.")
        return
    
    # Ask user which datasets to train
    print("\n" + "=" * 70)
    print("Select datasets to train:")
    print("1. Accident Detection")
    print("2. Garbage Detection")
    print("3. Violence Detection")
    print("4. Train All")
    print("=" * 70)
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    datasets_to_train = []
    if choice == '1':
        datasets_to_train = ['accident']
    elif choice == '2':
        datasets_to_train = ['garbage']
    elif choice == '3':
        datasets_to_train = ['violence']
    elif choice == '4':
        datasets_to_train = list(available_datasets.keys())
    else:
        print("❌ Invalid choice!")
        return
    
    # Train selected datasets
    print(f"\n🎯 Training {len(datasets_to_train)} dataset(s)...\n")
    
    results = {}
    for dataset_key in datasets_to_train:
        if dataset_key in available_datasets:
            success = train_model(dataset_key, available_datasets[dataset_key])
            results[dataset_key] = success
        else:
            print(f"\n⚠️ Skipping {dataset_key} - dataset not found")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TRAINING SUMMARY")
    print("=" * 70)
    for dataset_key, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {DATASETS[dataset_key]['name']}: {status}")
    print("=" * 70)
    
    print("\n🎉 All training jobs completed!")
    print("\n💡 Next Steps:")
    print("   1. Check training results in: runs/train/<model_name>/")
    print("   2. View metrics: runs/train/<model_name>/results.png")
    print("   3. Test models using: python test_models.py")
    print("   4. Export models: python export_models.py")

if __name__ == "__main__":
    main()
