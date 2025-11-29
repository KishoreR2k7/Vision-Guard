"""
Train Violence Detection Model
Large dataset: 60K images with NonViolence and Violence classes
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
from ultralytics import YOLO
import torch

def main():
    print("=" * 70)
    print("⚔️  VIOLENCE DETECTION TRAINING")
    print("=" * 70)
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("\n⚠️ Training on CPU (will be very slow)")
        return
    
    # Dataset info
    data_yaml = Path("dataset_prepared/data.yaml")
    
    print("\n📊 Dataset Information:")
    print("   Training images: 41,999")
    print("   Validation images: 8,999")
    print("   Test images: 9,001")
    print("   Classes: NonViolence, Violence")
    print(f"   Data config: {data_yaml}")
    
    # Training configuration
    print("\n⚙️ Training Configuration:")
    print("   Model: YOLOv11-Nano")
    print("   Epochs: 50 (adjustable)")
    print("   Batch Size: 16")
    print("   Image Size: 640")
    print("   Early Stopping: Patience 15")
    print("   Expected time: ~4-6 hours on RTX 3050")
    
    # Confirm
    print("\n" + "=" * 70)
    response = input("Start training? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("\n⏸️ Training cancelled.")
        print("\n💡 When ready, run: python train_violence.py")
        return
    
    # Train
    print("\n🚀 Starting violence detection training...\n")
    print("This will take several hours. You can monitor progress below.")
    print("Press Ctrl+C to stop training early (model will be saved).\n")
    
    try:
        model = YOLO("yolo11n.pt")
        
        results = model.train(
            data=str(data_yaml),
            epochs=50,  # Reduced from 100 due to large dataset
            batch=16,
            imgsz=640,
            device=0,
            project='runs/train',
            name='violence_detection',
            patience=15,  # Early stopping
            save=True,
            save_period=5,  # Save checkpoint every 5 epochs
            pretrained=True,
            optimizer='AdamW',
            lr0=0.01,
            amp=True,
            verbose=True,
            workers=4,  # More workers for large dataset
            
            # Optimized for large dataset
            cache=False,  # Don't cache (too much data)
            close_mosaic=10,
        )
        
        print("\n" + "=" * 70)
        print("✅ VIOLENCE DETECTION TRAINING COMPLETE!")
        print("=" * 70)
        print("\n📁 Results: runs/train/violence_detection/")
        print("🎯 Best model: runs/train/violence_detection/weights/best.pt")
        print("\n📊 View training curves:")
        print("   - results.png")
        print("   - confusion_matrix.png")
        print("\n🧪 Test the model:")
        print("   from ultralytics import YOLO")
        print("   model = YOLO('runs/train/violence_detection/weights/best.pt')")
        print("   results = model('test_video.mp4')")
        
    except KeyboardInterrupt:
        print("\n\n⏸️ Training interrupted by user.")
        print("📁 Partial model saved in: runs/train/violence_detection/weights/last.pt")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()
