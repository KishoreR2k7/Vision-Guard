"""
Train All Models - Accident, Garbage, and Violence Detection
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from pathlib import Path
from ultralytics import YOLO
import torch
import yaml
from datetime import datetime

def train_model(model_name, data_yaml, epochs, description):
    """Train a single model"""
    
    print("\n" + "=" * 70)
    print(f"🚀 Training: {model_name.upper()}")
    print("=" * 70)
    print(f"📝 {description}")
    
    try:
        model = YOLO("yolo11n.pt")
        
        start_time = datetime.now()
        
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            batch=16,
            imgsz=640,
            device=0 if torch.cuda.is_available() else 'cpu',
            project='runs/train',
            name=model_name,
            patience=20 if epochs <= 50 else 15,
            save=True,
            save_period=10,
            pretrained=True,
            optimizer='AdamW',
            lr0=0.01,
            amp=True,
            verbose=True,
            workers=2 if epochs <= 50 else 4,
        )
        
        duration = datetime.now() - start_time
        
        print("\n" + "=" * 70)
        print(f"✅ {model_name.upper()} TRAINING COMPLETE!")
        print("=" * 70)
        print(f"⏱️  Training time: {duration}")
        print(f"📁 Results: runs/train/{model_name}/")
        print(f"🎯 Best model: runs/train/{model_name}/weights/best.pt")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error training {model_name}: {e}")
        return False

def main():
    print("=" * 70)
    print("🎯 VISION GUARD AI - COMPLETE TRAINING PIPELINE")
    print("=" * 70)
    
    # Check GPU
    if not torch.cuda.is_available():
        print("\n❌ No GPU detected! Training will be extremely slow.")
        print("Please ensure CUDA is properly installed.")
        return
    
    print(f"\n✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Define training jobs
    training_jobs = [
        {
            'name': 'accident_detection',
            'data_yaml': Path('dataset/accident_detetcion/data1/data.yaml'),
            'epochs': 100,
            'description': 'Accident Detection: 369 train, 46 val images',
            'estimated_time': '10 minutes'
        },
        {
            'name': 'garbage_detection',
            'data_yaml': Path('dataset/garbage/data1/data.yaml'),
            'epochs': 100,
            'description': 'Garbage Detection: Classification task',
            'estimated_time': '15 minutes'
        },
        {
            'name': 'violence_detection',
            'data_yaml': Path('dataset_prepared/data.yaml'),
            'epochs': 50,
            'description': 'Violence Detection: 42K train, 9K val images (2 classes)',
            'estimated_time': '4-6 hours'
        }
    ]
    
    # Display training plan
    print("\n📋 Training Plan:")
    print("-" * 70)
    total_time = "~5-7 hours"
    for i, job in enumerate(training_jobs, 1):
        status = "✅ Already trained" if Path(f"runs/train/{job['name']}/weights/best.pt").exists() else "⏳ Pending"
        print(f"{i}. {job['name'].replace('_', ' ').title()}")
        print(f"   {job['description']}")
        print(f"   Estimated time: {job['estimated_time']}")
        print(f"   Status: {status}")
        print()
    
    print(f"📊 Total estimated time: {total_time}")
    print("=" * 70)
    
    # User confirmation
    print("\nOptions:")
    print("1. Train all models")
    print("2. Train only pending models (skip completed)")
    print("3. Select specific models")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '4':
        print("\n👋 Training cancelled.")
        return
    
    # Determine which models to train
    models_to_train = []
    
    if choice == '1':
        models_to_train = training_jobs
        
    elif choice == '2':
        models_to_train = [
            job for job in training_jobs 
            if not Path(f"runs/train/{job['name']}/weights/best.pt").exists()
        ]
        
    elif choice == '3':
        print("\nSelect models to train:")
        for i, job in enumerate(training_jobs, 1):
            print(f"{i}. {job['name'].replace('_', ' ').title()}")
        
        selections = input("\nEnter numbers (e.g., 1,3 or 1-3): ").strip()
        
        try:
            if '-' in selections:
                start, end = map(int, selections.split('-'))
                models_to_train = training_jobs[start-1:end]
            else:
                indices = [int(x.strip())-1 for x in selections.split(',')]
                models_to_train = [training_jobs[i] for i in indices]
        except:
            print("\n❌ Invalid selection!")
            return
    
    if not models_to_train:
        print("\n✅ All selected models are already trained!")
        return
    
    # Start training
    print(f"\n🚀 Starting training for {len(models_to_train)} model(s)...")
    print("=" * 70)
    
    results = {}
    for job in models_to_train:
        success = train_model(
            job['name'],
            job['data_yaml'],
            job['epochs'],
            job['description']
        )
        results[job['name']] = success
    
    # Final summary
    print("\n" + "=" * 70)
    print("📊 TRAINING SUMMARY")
    print("=" * 70)
    
    for model_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {model_name.replace('_', ' ').title()}: {status}")
    
    print("\n" + "=" * 70)
    print("🎉 All training jobs completed!")
    print("\n📁 Models saved in: runs/train/")
    print("\n💡 Next steps:")
    print("   1. Test models with: python test_all_models.py")
    print("   2. View results in runs/train/<model_name>/")
    print("   3. Export for production: model.export(format='onnx')")

if __name__ == "__main__":
    main()
