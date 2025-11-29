"""
Dataset Preparation Script for Violence Detection
This script prepares the video dataset for training by:
1. Extracting frames from videos
2. Creating train/val/test splits
3. Preparing data.yaml for YOLO training
"""

import cv2
import os
import yaml
import shutil
from pathlib import Path
import random
from tqdm import tqdm

# Configuration
DATASET_ROOT = Path("../dataset")
OUTPUT_ROOT = Path("../dataset_prepared")
VIOLENCE_DIR = DATASET_ROOT / "Violence"
NON_VIOLENCE_DIR = DATASET_ROOT / "NonViolence"

# Frame extraction settings
FRAMES_PER_VIDEO = 30  # Number of frames to extract from each video
SKIP_FRAMES = 5  # Extract every Nth frame

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# YOLO classes
CLASSES = ["NonViolence", "Violence"]


def extract_frames_from_video(video_path, output_dir, class_id, max_frames=FRAMES_PER_VIDEO):
    """Extract frames from a video file"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return 0
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // max_frames) if total_frames > max_frames else 1
    
    frames_extracted = 0
    frame_idx = 0
    
    while frames_extracted < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            # Save frame
            video_name = video_path.stem
            frame_filename = f"{video_name}_frame_{frames_extracted:04d}.jpg"
            frame_path = output_dir / frame_filename
            cv2.imwrite(str(frame_path), frame)
            
            # Create YOLO format label file
            label_filename = f"{video_name}_frame_{frames_extracted:04d}.txt"
            label_path = output_dir / label_filename
            
            # Full frame annotation (class_id x_center y_center width height)
            # For video-level labels, we annotate the entire frame
            with open(label_path, 'w') as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")
            
            frames_extracted += 1
        
        frame_idx += 1
    
    cap.release()
    return frames_extracted


def prepare_yolo_dataset():
    """Prepare dataset in YOLO format"""
    print("=" * 60)
    print("Preparing Violence Detection Dataset for YOLO Training")
    print("=" * 60)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        (OUTPUT_ROOT / 'images' / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Process Violence videos (class 1)
    print("\n[1/2] Processing Violence videos...")
    violence_videos = list(VIOLENCE_DIR.glob("*.mp4")) + list(VIOLENCE_DIR.glob("*.avi"))
    print(f"Found {len(violence_videos)} violence videos")
    
    temp_violence_dir = OUTPUT_ROOT / 'temp_violence'
    temp_violence_dir.mkdir(parents=True, exist_ok=True)
    
    for video_path in tqdm(violence_videos, desc="Extracting Violence frames"):
        extract_frames_from_video(video_path, temp_violence_dir, class_id=1)
    
    # Process NonViolence videos (class 0)
    print("\n[2/2] Processing NonViolence videos...")
    non_violence_videos = list(NON_VIOLENCE_DIR.glob("*.mp4")) + list(NON_VIOLENCE_DIR.glob("*.avi"))
    print(f"Found {len(non_violence_videos)} non-violence videos")
    
    temp_non_violence_dir = OUTPUT_ROOT / 'temp_non_violence'
    temp_non_violence_dir.mkdir(parents=True, exist_ok=True)
    
    for video_path in tqdm(non_violence_videos, desc="Extracting NonViolence frames"):
        extract_frames_from_video(video_path, temp_non_violence_dir, class_id=0)
    
    # Get all extracted frames
    violence_frames = list(temp_violence_dir.glob("*.jpg"))
    non_violence_frames = list(temp_non_violence_dir.glob("*.jpg"))
    
    print(f"\nExtracted {len(violence_frames)} violence frames")
    print(f"Extracted {len(non_violence_frames)} non-violence frames")
    
    # Combine and shuffle
    all_frames = violence_frames + non_violence_frames
    random.shuffle(all_frames)
    
    # Split dataset
    total_frames = len(all_frames)
    train_end = int(total_frames * TRAIN_RATIO)
    val_end = train_end + int(total_frames * VAL_RATIO)
    
    train_frames = all_frames[:train_end]
    val_frames = all_frames[train_end:val_end]
    test_frames = all_frames[val_end:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_frames)} frames ({TRAIN_RATIO*100:.0f}%)")
    print(f"  Val:   {len(val_frames)} frames ({VAL_RATIO*100:.0f}%)")
    print(f"  Test:  {len(test_frames)} frames ({TEST_RATIO*100:.0f}%)")
    
    # Copy files to appropriate directories
    def copy_frames(frames, split):
        for frame_path in tqdm(frames, desc=f"Copying {split} set"):
            # Copy image
            shutil.copy2(frame_path, OUTPUT_ROOT / 'images' / split / frame_path.name)
            
            # Copy label
            label_path = frame_path.with_suffix('.txt')
            if label_path.exists():
                shutil.copy2(label_path, OUTPUT_ROOT / 'labels' / split / label_path.name)
    
    print("\nCopying files to train/val/test directories...")
    copy_frames(train_frames, 'train')
    copy_frames(val_frames, 'val')
    copy_frames(test_frames, 'test')
    
    # Clean up temporary directories
    print("\nCleaning up temporary files...")
    shutil.rmtree(temp_violence_dir)
    shutil.rmtree(temp_non_violence_dir)
    
    # Create data.yaml for YOLO training
    data_yaml = {
        'path': str(OUTPUT_ROOT.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': 2,
        'names': CLASSES
    }
    
    yaml_path = OUTPUT_ROOT / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"\n✅ Dataset preparation complete!")
    print(f"📁 Output directory: {OUTPUT_ROOT.absolute()}")
    print(f"📄 YOLO config file: {yaml_path.absolute()}")
    print(f"\nYou can now train YOLO with this dataset!")


if __name__ == "__main__":
    random.seed(42)  # For reproducibility
    prepare_yolo_dataset()
