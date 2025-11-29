"""
Data Preparation for YOLO + LSTM Training
This script prepares sequences of frames from videos for training the YOLO + LSTM model
"""

import cv2
import os
import json
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm
import pickle

# Configuration
DATASET_ROOT = Path("../dataset")
OUTPUT_ROOT = Path("../dataset_sequences")
VIOLENCE_DIR = DATASET_ROOT / "Violence"
NON_VIOLENCE_DIR = DATASET_ROOT / "NonViolence"

# Sequence settings (optimized for RTX 3050 6GB)
SEQUENCE_LENGTH = 12  # Reduced from 16 for faster training
FRAME_SIZE = (160, 160)  # Smaller size for faster processing
STRIDE = 10  # Larger stride = fewer sequences = faster training

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def extract_sequences_from_video(video_path, label, sequence_length=SEQUENCE_LENGTH):
    """Extract sequences of frames from a video"""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    
    sequences = []
    frames_buffer = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Resize frame
        frame = cv2.resize(frame, FRAME_SIZE)
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalize
        frame = frame.astype(np.float32) / 255.0
        
        frames_buffer.append(frame)
        
        # Create sequence when buffer is full
        if len(frames_buffer) == sequence_length:
            sequences.append({
                'frames': np.array(frames_buffer),
                'label': label
            })
            # Slide window with stride
            frames_buffer = frames_buffer[STRIDE:]
    
    cap.release()
    return sequences


def prepare_lstm_dataset():
    """Prepare sequences dataset for LSTM training"""
    print("=" * 60)
    print("Preparing Video Sequences for YOLO + LSTM Training")
    print("=" * 60)
    
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    
    all_sequences = []
    
    # Process Violence videos (label 1)
    print("\n[1/2] Processing Violence videos...")
    violence_videos = list(VIOLENCE_DIR.glob("*.mp4")) + list(VIOLENCE_DIR.glob("*.avi"))
    print(f"Found {len(violence_videos)} violence videos")
    
    for video_path in tqdm(violence_videos, desc="Extracting Violence sequences"):
        sequences = extract_sequences_from_video(video_path, label=1)
        all_sequences.extend(sequences)
    
    violence_count = len(all_sequences)
    
    # Process NonViolence videos (label 0)
    print("\n[2/2] Processing NonViolence videos...")
    non_violence_videos = list(NON_VIOLENCE_DIR.glob("*.mp4")) + list(NON_VIOLENCE_DIR.glob("*.avi"))
    print(f"Found {len(non_violence_videos)} non-violence videos")
    
    for video_path in tqdm(non_violence_videos, desc="Extracting NonViolence sequences"):
        sequences = extract_sequences_from_video(video_path, label=0)
        all_sequences.extend(sequences)
    
    non_violence_count = len(all_sequences) - violence_count
    
    print(f"\nExtracted sequences:")
    print(f"  Violence: {violence_count} sequences")
    print(f"  NonViolence: {non_violence_count} sequences")
    print(f"  Total: {len(all_sequences)} sequences")
    
    # Shuffle
    random.shuffle(all_sequences)
    
    # Split dataset
    total_sequences = len(all_sequences)
    train_end = int(total_sequences * TRAIN_RATIO)
    val_end = train_end + int(total_sequences * VAL_RATIO)
    
    train_sequences = all_sequences[:train_end]
    val_sequences = all_sequences[train_end:val_end]
    test_sequences = all_sequences[val_end:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_sequences)} sequences ({TRAIN_RATIO*100:.0f}%)")
    print(f"  Val:   {len(val_sequences)} sequences ({VAL_RATIO*100:.0f}%)")
    print(f"  Test:  {len(test_sequences)} sequences ({TEST_RATIO*100:.0f}%)")
    
    # Save sequences
    def save_sequences(sequences, split):
        output_file = OUTPUT_ROOT / f"{split}_sequences.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(sequences, f)
        print(f"Saved {len(sequences)} {split} sequences to {output_file}")
    
    print("\nSaving sequences...")
    save_sequences(train_sequences, 'train')
    save_sequences(val_sequences, 'val')
    save_sequences(test_sequences, 'test')
    
    # Save metadata
    metadata = {
        'sequence_length': SEQUENCE_LENGTH,
        'frame_size': FRAME_SIZE,
        'num_classes': 2,
        'classes': ['NonViolence', 'Violence'],
        'train_size': len(train_sequences),
        'val_size': len(val_sequences),
        'test_size': len(test_sequences)
    }
    
    metadata_file = OUTPUT_ROOT / 'metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)
    
    print(f"\n✅ Sequence preparation complete!")
    print(f"📁 Output directory: {OUTPUT_ROOT.absolute()}")
    print(f"📄 Metadata file: {metadata_file.absolute()}")


if __name__ == "__main__":
    random.seed(42)
    prepare_lstm_dataset()
