"""
Master Training Script - Train All Models
This script runs the complete training pipeline for violence detection
"""

import subprocess
import sys
from pathlib import Path
import argparse


def run_command(command, description):
    """Run a command and handle errors"""
    print("\n" + "=" * 70)
    print(f"🚀 {description}")
    print("=" * 70)
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            text=True,
            capture_output=False
        )
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        return False


def train_all():
    """Train both YOLO and YOLO+LSTM models"""
    
    print("\n" + "=" * 70)
    print("🎯 Violence Detection - Complete Training Pipeline")
    print("=" * 70)
    print("\nThis will train both:")
    print("  1. YOLO frame-level violence detector")
    print("  2. YOLO + LSTM temporal video classifier")
    print("\n⏱️  Estimated time: 3-6 hours (depends on GPU)")
    print("💾 Required space: ~10GB for processed datasets and models")
    
    # Confirm
    response = input("\nProceed with training? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        return
    
    # Step 1: Prepare YOLO dataset
    print("\n\n📦 STEP 1/4: Preparing YOLO dataset...")
    if not run_command("python prepare_dataset.py", "Dataset preparation"):
        print("\n❌ Failed at dataset preparation. Exiting...")
        return
    
    # Step 2: Train YOLO
    print("\n\n🔥 STEP 2/4: Training YOLO model...")
    if not run_command("python train_yolo.py", "YOLO training"):
        print("\n⚠️  YOLO training failed, but continuing with LSTM...")
    
    # Step 3: Prepare sequences for LSTM
    print("\n\n📦 STEP 3/4: Preparing video sequences for LSTM...")
    if not run_command("python prepare_sequences.py", "Sequence preparation"):
        print("\n❌ Failed at sequence preparation. Exiting...")
        return
    
    # Step 4: Train LSTM
    print("\n\n🔥 STEP 4/4: Training YOLO + LSTM model...")
    if not run_command("python train_yolo_lstm.py", "YOLO + LSTM training"):
        print("\n⚠️  LSTM training failed.")
    
    # Summary
    print("\n\n" + "=" * 70)
    print("🎉 TRAINING PIPELINE COMPLETED!")
    print("=" * 70)
    
    # Check outputs
    yolo_model = Path("violence_detection/yolo11n_violence/weights/best.onnx")
    lstm_model = Path("models/yolo_lstm/best_model.pth")
    
    print("\n📊 Training Results:")
    print(f"  YOLO Model: {'✅ Found' if yolo_model.exists() else '❌ Not found'}")
    if yolo_model.exists():
        print(f"    Location: {yolo_model.absolute()}")
    
    print(f"  LSTM Model: {'✅ Found' if lstm_model.exists() else '❌ Not found'}")
    if lstm_model.exists():
        print(f"    Location: {lstm_model.absolute()}")
    
    print("\n💡 Next Steps:")
    print("  1. Test YOLO model: python ../vision/yolov11_pipeline.py")
    print("  2. Test LSTM model: python ../vision/violence_lstm_inference.py --video <path>")
    print("  3. Review training metrics in the output directories")
    print("\n✨ Happy detecting!")


def train_yolo_only():
    """Train only YOLO model"""
    print("\n🎯 Training YOLO model only...")
    
    if not run_command("python prepare_dataset.py", "Dataset preparation"):
        return
    
    if not run_command("python train_yolo.py", "YOLO training"):
        return
    
    print("\n✅ YOLO training completed!")


def train_lstm_only():
    """Train only LSTM model"""
    print("\n🎯 Training YOLO + LSTM model only...")
    
    if not run_command("python prepare_sequences.py", "Sequence preparation"):
        return
    
    if not run_command("python train_yolo_lstm.py", "YOLO + LSTM training"):
        return
    
    print("\n✅ LSTM training completed!")


def main():
    parser = argparse.ArgumentParser(
        description='Master training script for violence detection models'
    )
    parser.add_argument(
        '--mode',
        type=str,
        choices=['all', 'yolo', 'lstm'],
        default='all',
        help='Training mode: all (both models), yolo (YOLO only), lstm (LSTM only)'
    )
    
    args = parser.parse_args()
    
    # Change to training directory
    training_dir = Path(__file__).parent
    import os
    os.chdir(training_dir)
    
    if args.mode == 'all':
        train_all()
    elif args.mode == 'yolo':
        train_yolo_only()
    elif args.mode == 'lstm':
        train_lstm_only()


if __name__ == "__main__":
    main()
