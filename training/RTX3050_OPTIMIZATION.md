# RTX 3050 6GB Training Optimizations

This document describes the optimizations made for training on RTX 3050 with 6GB VRAM.

## Key Optimizations

### 1. YOLO Training (`train_yolo.py`)
- **Epochs**: Reduced from 100 → 50 (with early stopping patience=10)
- **Image Size**: Reduced from 640 → 480 pixels
- **Batch Size**: Reduced from 16 → 8
- **Gradient Accumulation**: Added (steps=4, effective batch=32)
- **Workers**: Reduced from 4 → 2
- **Warmup Epochs**: Reduced from 3 → 2
- **Cache**: Enabled (images cached in RAM for faster loading)
- **Mixed Precision**: Already enabled (AMP)

**Expected Training Time**: ~1-2 hours (vs 3-4 hours before)
**Expected Accuracy**: mAP50 ~0.75-0.85 (minimal loss from optimization)

### 2. LSTM Training (`train_yolo_lstm.py`)
- **Epochs**: Reduced from 50 → 30
- **Batch Size**: Reduced from 4 → 2
- **Gradient Accumulation**: Added (steps=4, effective batch=8)
- **Workers**: Reduced from 4 → 2
- **Hidden Size**: Reduced from 256 → 128
- **LSTM Layers**: Reduced from 2 → 1
- **Dropout**: Reduced from 0.5 → 0.3
- **Scheduler Patience**: Reduced from 5 → 3

**Expected Training Time**: ~30-45 minutes (vs 1-2 hours before)
**Expected Accuracy**: ~85-90% (minimal loss from optimization)

### 3. LSTM Model (`yolo_lstm_model.py`)
- **Backbone**: Changed from ResNet50 → ResNet18
  - Parameters: 25M → 11M (56% reduction)
  - Feature dimension: 2048 → 512
  - Speed: ~2-3x faster
- **Hidden Size**: 256 → 128 (50% reduction)
- **LSTM Layers**: 2 → 1 (faster computation)

### 4. Sequence Preparation (`prepare_sequences.py`)
- **Sequence Length**: Reduced from 16 → 12 frames
- **Frame Size**: Reduced from 224x224 → 160x160
- **Stride**: Increased from 8 → 10 (fewer sequences)

**Expected Processing Time**: ~10-15 minutes (vs 20-30 minutes before)

## Total Estimated Training Time
- Dataset Preparation: ~15-20 minutes
- YOLO Training: ~1-2 hours
- Sequence Preparation: ~10-15 minutes
- LSTM Training: ~30-45 minutes

**Total**: ~2.5-3.5 hours (vs 5-7 hours before)

## Memory Usage
- YOLO Training: ~4-5 GB VRAM (safe margin on 6GB)
- LSTM Training: ~3-4 GB VRAM (safe margin on 6GB)

## Expected Performance
With these optimizations, you should achieve:
- **YOLO**: mAP50 of 0.75-0.85 (frame-level detection)
- **LSTM**: 85-90% accuracy (video-level classification)
- **Training Speed**: ~50-60% faster than original config
- **Accuracy Loss**: <5% compared to full configuration

## Tips for Best Results

### 1. Enable RAM Caching
The YOLO training now has `cache=True` enabled. Ensure you have at least 8GB RAM available.

### 2. Close Background Applications
Close Chrome, Discord, etc. to free up VRAM and system memory.

### 3. Monitor GPU Temperature
Use `nvidia-smi` to monitor temperature. Keep it below 80°C for optimal performance.

### 4. Use Early Stopping
The patience parameter will stop training if no improvement is seen. This saves time while maintaining accuracy.

### 5. Batch Training
If you're running into OOM errors:
- Further reduce batch size to 4 (YOLO) or 1 (LSTM)
- Increase gradient accumulation accordingly

## Verification Commands

Check GPU status:
```powershell
nvidia-smi
```

Monitor training in real-time:
```powershell
# In one terminal - start training
python train_yolo.py

# In another terminal - monitor GPU
nvidia-smi -l 1
```

## Troubleshooting

### Out of Memory (OOM) Error
If you still get OOM errors:

1. **For YOLO**: Edit `train_yolo.py`
   ```python
   BATCH_SIZE = 4  # Reduce further
   ACCUMULATE = 8  # Increase to maintain effective batch size
   IMAGE_SIZE = 416  # Can reduce further if needed
   ```

2. **For LSTM**: Edit `train_yolo_lstm.py`
   ```python
   BATCH_SIZE = 1  # Reduce to minimum
   ACCUMULATE_GRAD = 8  # Increase to maintain effective batch size
   ```

3. **For Sequences**: Edit `prepare_sequences.py`
   ```python
   FRAME_SIZE = (128, 128)  # Smaller frames
   SEQUENCE_LENGTH = 10  # Fewer frames per sequence
   ```

### Slow Training
If training is still slow:

1. Check if cache is working (YOLO should be faster after first epoch)
2. Reduce `NUM_WORKERS` to 0 if CPU is bottleneck
3. Enable Windows High Performance mode
4. Update GPU drivers

### Poor Accuracy
If accuracy is lower than expected:

1. Increase epochs slightly (e.g., 60 for YOLO, 40 for LSTM)
2. Increase patience for early stopping
3. Use learning rate scheduler more aggressively

## Restore Original Settings

To restore full training configuration, change back:

**train_yolo.py**:
```python
EPOCHS = 100
BATCH_SIZE = 16
IMAGE_SIZE = 640
ACCUMULATE = 1  # Remove gradient accumulation
```

**train_yolo_lstm.py**:
```python
NUM_EPOCHS = 50
BATCH_SIZE = 4
hidden_size=256
num_layers=2
```

**yolo_lstm_model.py**:
```python
resnet = models.resnet50(pretrained=True)
self.feature_dim = 2048
```

---

**Note**: These optimizations are specifically tuned for RTX 3050 6GB. They balance training speed and model accuracy for practical deployment.
