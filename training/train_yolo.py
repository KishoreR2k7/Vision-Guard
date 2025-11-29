"""
YOLOv11 Training Script for Violence Detection
This script trains a YOLOv11 model to detect violence in video frames
"""

import os
from pathlib import Path
from ultralytics import YOLO
import torch

# Configuration
DATA_YAML = Path("../dataset_prepared/data.yaml")
MODEL_SIZE = "yolo11n.pt"  # Options: yolo11n, yolo11s, yolo11m, yolo11l, yolo11x
EPOCHS = 50  # Reduced for faster training with early stopping
BATCH_SIZE = 16  # Optimized for 6GB VRAM with mixed precision
IMAGE_SIZE = 480  # Reduced from 640 for faster training
DEVICE = 0 if torch.cuda.is_available() else 'cpu'
PROJECT_NAME = "violence_detection"
EXPERIMENT_NAME = "yolo11n_violence_optimized"

# Training hyperparameters optimized for RTX 3050 6GB
LEARNING_RATE = 0.01
PATIENCE = 10  # Early stopping patience (reduced)
SAVE_PERIOD = 10  # Save checkpoint every N epochs
CLOSE_MOSAIC = 10  # Close mosaic augmentation in last N epochs for better accuracy


def train_yolo():
    """Train YOLOv11 model for violence detection"""
    
    print("=" * 60)
    print("Training YOLOv11 for Violence Detection")
    print("=" * 60)
    
    # Check if data.yaml exists
    if not DATA_YAML.exists():
        print(f"❌ Error: {DATA_YAML} not found!")
        print("Please run prepare_dataset.py first to prepare the dataset.")
        return
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️ No GPU available, training on CPU (will be slower)")
    
    print(f"\n📊 Training Configuration (RTX 3050 6GB Optimized):")
    print(f"   Model: {MODEL_SIZE}")
    print(f"   Epochs: {EPOCHS}")
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Image Size: {IMAGE_SIZE}")
    print(f"   Device: {DEVICE}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    print(f"   Mixed Precision: Enabled (AMP)")
    print(f"   Close Mosaic: Last {CLOSE_MOSAIC} epochs")
    
    # Load pretrained YOLO model
    print(f"\n🔄 Loading {MODEL_SIZE} model...")
    model = YOLO(MODEL_SIZE)
    
    # Train the model
    print("\n🚀 Starting training...")
    print("-" * 60)
    
    try:
        results = model.train(
            data=str(DATA_YAML),
            epochs=EPOCHS,
            batch=BATCH_SIZE,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            project=PROJECT_NAME,
            name=EXPERIMENT_NAME,
            lr0=LEARNING_RATE,
            lrf=0.01,  # Final learning rate (lr0 * lrf)
            momentum=0.937,
            weight_decay=0.0005,
            warmup_epochs=2.0,  # Reduced warmup
            warmup_momentum=0.8,
            warmup_bias_lr=0.1,
            patience=PATIENCE,
            save_period=SAVE_PERIOD,
            pretrained=True,
            optimizer='AdamW',
            verbose=True,
            
            # Performance optimization for RTX 3050 6GB
            amp=True,  # Enable Automatic Mixed Precision for faster training
            
            # Data augmentation (optimized for violence detection)
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10.0,
            translate=0.1,
            scale=0.5,
            shear=0.0,
            perspective=0.0,
            flipud=0.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.15,  # Increased for better generalization
            copy_paste=0.0,
            close_mosaic=CLOSE_MOSAIC,  # Close mosaic augmentation in last N epochs
            
            # Other settings
            workers=2,  # Reduced for RTX 3050 6GB
            exist_ok=True,
            resume=False,
            cache=True,  # Cache images in RAM for faster training
        )
        
        print("\n" + "=" * 60)
        print("✅ Training completed successfully!")
        print("=" * 60)
        
        # Print model location
        best_model = Path(PROJECT_NAME) / EXPERIMENT_NAME / "weights" / "best.pt"
        last_model = Path(PROJECT_NAME) / EXPERIMENT_NAME / "weights" / "last.pt"
        
        print(f"\n📁 Model weights saved:")
        print(f"   Best model:  {best_model.absolute()}")
        print(f"   Last model:  {last_model.absolute()}")
        
        # Validate the model
        print("\n📊 Validating best model...")
        metrics = model.val(data=str(DATA_YAML))
        
        print(f"\n📈 Validation Metrics:")
        print(f"   mAP50:     {metrics.box.map50:.4f}")
        print(f"   mAP50-95:  {metrics.box.map:.4f}")
        print(f"   Precision: {metrics.box.mp:.4f}")
        print(f"   Recall:    {metrics.box.mr:.4f}")
        
        # Export to ONNX for inference
        print("\n🔄 Exporting model to ONNX format...")
        onnx_path = model.export(format='onnx', imgsz=IMAGE_SIZE)
        print(f"✅ ONNX model saved: {onnx_path}")
        
        print("\n" + "=" * 60)
        print("🎉 All done! Your violence detection model is ready!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        raise


if __name__ == "__main__":
    train_yolo()
