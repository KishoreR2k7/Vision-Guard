"""
Dataset Preparation Helper - Creates YOLO labels for image-only datasets
For datasets that have images but no bounding box labels (full-frame classification)
"""

import os
from pathlib import Path
import yaml

def create_full_frame_labels(dataset_path, class_name='object', class_id=0):
    """
    Create YOLO labels that cover the full frame (for classification tasks)
    Format: class_id x_center y_center width height (normalized 0-1)
    Full frame: 0 0.5 0.5 1.0 1.0
    """
    
    print(f"\n📝 Creating full-frame labels for: {dataset_path}")
    
    splits = ['train', 'val', 'test']
    total_created = 0
    
    for split in splits:
        # Check for images in split/class_name structure
        image_dir = dataset_path / split / class_name
        
        if not image_dir.exists():
            # Try split directory directly
            image_dir = dataset_path / split
            if not image_dir.exists():
                print(f"   ⚠️ Skipping {split} - directory not found")
                continue
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_files = [f for f in image_dir.iterdir() if f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"   ⚠️ No images found in {split}")
            continue
        
        print(f"   Processing {split}: {len(image_files)} images")
        
        # Create labels for each image
        for img_file in image_files:
            label_file = img_file.with_suffix('.txt')
            
            if not label_file.exists():
                # Full frame bounding box: class_id x_center y_center width height
                # All normalized (0-1): center at (0.5, 0.5), width=1.0, height=1.0
                with open(label_file, 'w') as f:
                    f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
                total_created += 1
        
        print(f"   ✅ Created {len(image_files)} labels in {split}")
    
    print(f"\n✅ Total labels created: {total_created}")
    return total_created

def create_accident_dataset_yaml():
    """Create data.yaml for accident detection dataset"""
    
    dataset_path = Path("../dataset/accident_detetcion/data1")
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return False
    
    yaml_path = dataset_path / "data.yaml"
    
    # Check structure
    has_accident_subdir = (dataset_path / 'train' / 'Accident').exists()
    
    data = {
        'path': str(dataset_path.absolute()),
        'train': 'train/Accident' if has_accident_subdir else 'train',
        'val': 'val/Accident' if has_accident_subdir else 'val',
        'test': 'test/Accident' if has_accident_subdir else 'test',
        'nc': 1,
        'names': ['Accident']
    }
    
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    
    print(f"✅ Created {yaml_path}")
    return True

def create_garbage_dataset_yaml():
    """Fix data.yaml for garbage detection dataset"""
    
    dataset_path = Path("../dataset/garbage/data1")
    yaml_path = dataset_path / "data.yaml"
    
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        return False
    
    # Update the yaml to use correct paths
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
    
    print(f"✅ Updated {yaml_path}")
    return True

def main():
    print("=" * 70)
    print("🔧 DATASET PREPARATION HELPER")
    print("=" * 70)
    
    print("\n1️⃣ Preparing Accident Detection Dataset...")
    accident_path = Path("../dataset/accident_detetcion/data1")
    if accident_path.exists():
        create_accident_dataset_yaml()
        # Create full-frame labels for accident detection
        create_full_frame_labels(accident_path, class_name='Accident', class_id=0)
    else:
        print("   ❌ Accident dataset not found")
    
    print("\n2️⃣ Preparing Garbage Detection Dataset...")
    garbage_path = Path("../dataset/garbage/data1")
    if garbage_path.exists():
        create_garbage_dataset_yaml()
        print("   ✅ Garbage dataset already has labels")
    else:
        print("   ❌ Garbage dataset not found")
    
    print("\n" + "=" * 70)
    print("✅ Dataset preparation complete!")
    print("\n💡 Next step: Run training script")
    print("   python train_multi_class.py")
    print("=" * 70)

if __name__ == "__main__":
    main()
