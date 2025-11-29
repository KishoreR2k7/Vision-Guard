# Violence Detection - Quick Start Guide 🚀

## Overview

Your Vision Guard AI system now includes violence detection capabilities using two approaches:

1. **YOLO Frame-level Detection** - Fast, real-time detection in video frames
2. **YOLO + LSTM Temporal Detection** - Accurate video-level classification with temporal understanding

## 📋 Prerequisites

```bash
# Install all required dependencies
pip install -r requirements.txt
```

## 🎯 Training Your Models

### Quick Training (All Models)

Run everything with a single command:

```bash
cd training
python train_all.py
```

This will:
- ✅ Prepare datasets (extract frames and sequences)
- ✅ Train YOLO violence detection model (~2-3 hours)
- ✅ Train YOLO + LSTM model (~1-2 hours)

### Train Individual Models

**Option 1: YOLO Only (Frame Detection)**
```bash
cd training
python train_all.py --mode yolo
```

**Option 2: LSTM Only (Video Classification)**
```bash
cd training
python train_all.py --mode lstm
```

### Manual Step-by-Step Training

**For YOLO:**
```bash
cd training
python prepare_dataset.py   # Prepare frame dataset
python train_yolo.py         # Train YOLO model
```

**For YOLO + LSTM:**
```bash
cd training
python prepare_sequences.py  # Prepare video sequences
python train_yolo_lstm.py    # Train LSTM model
```

## 🔍 Using the Trained Models

### 1. Real-time Stream Processing (YOLO)

The main pipeline automatically uses the violence detection model:

```bash
python vision/yolov11_pipeline.py
```

The pipeline will:
- ✅ Auto-detect if violence model exists
- ✅ Use violence model for detection
- ✅ Publish incidents to Redis when violence detected

### 2. Video File Analysis (YOLO + LSTM)

Analyze individual video files:

```bash
# Basic usage
python vision/violence_lstm_inference.py --video path/to/video.mp4

# Save output video
python vision/violence_lstm_inference.py --video path/to/video.mp4 --output result.mp4

# No display (faster processing)
python vision/violence_lstm_inference.py --video path/to/video.mp4 --no-display

# Test on sample videos
python vision/violence_lstm_inference.py --video dataset/Violence/V_1.mp4 --output violence_test.mp4
python vision/violence_lstm_inference.py --video dataset/NonViolence/NV_1.mp4 --output nonviolence_test.mp4
```

## 📊 Understanding the Output

### YOLO Model Output

After training, you'll find:
```
training/violence_detection/yolo11n_violence/
├── weights/
│   ├── best.pt          # Best PyTorch weights
│   ├── best.onnx        # ONNX for inference (used by pipeline)
│   └── last.pt          # Last checkpoint
├── results.csv          # Training metrics
└── confusion_matrix.png # Validation results
```

### LSTM Model Output

After training, you'll find:
```
training/models/yolo_lstm/
├── best_model.pth              # Best model checkpoint
├── final_model.pth             # Final model
├── training_history.json       # All metrics
├── training_curves.png         # Loss/accuracy plots
└── confusion_matrix_best.png   # Performance visualization
```

## 📈 Model Performance

Expected performance after training:

| Model | Accuracy | Speed | Use Case |
|-------|----------|-------|----------|
| YOLO | ~85-90% | ~30 FPS | Real-time monitoring |
| YOLO+LSTM | ~90-95% | ~10 FPS | Video analysis |

## 🛠️ Configuration

### Adjust Training Parameters

Edit the training scripts to customize:

**In `train_yolo.py`:**
```python
EPOCHS = 100          # Number of training epochs
BATCH_SIZE = 16       # Batch size (reduce if OOM)
IMAGE_SIZE = 640      # Input image size
LEARNING_RATE = 0.01  # Learning rate
```

**In `train_yolo_lstm.py`:**
```python
BATCH_SIZE = 4        # Batch size (reduce if OOM)
NUM_EPOCHS = 50       # Number of training epochs
LEARNING_RATE = 0.0001 # Learning rate
```

### GPU Memory Issues?

Reduce batch size in the training scripts:
- YOLO: Change `BATCH_SIZE = 16` to `BATCH_SIZE = 8` or lower
- LSTM: Change `BATCH_SIZE = 4` to `BATCH_SIZE = 2` or `BATCH_SIZE = 1`

## 🔄 Integration with Main System

### Update Config

Edit `config.py` to use the violence detection model:

```python
# Change from default COCO model to violence model
yolo_model_path = "./training/violence_detection/yolo11n_violence/weights/best.onnx"
```

### Redis Event Format

Violence detection events published to Redis:

```json
{
  "id": "incident_20251121120500123",
  "incident": "violence_detected",
  "confidence": 0.95,
  "frames": ["url1", "url2", "url3"],
  "timestamp": "2025-11-21T12:05:00.123456",
  "location": {
    "lat": 13.0827,
    "lon": 80.2707
  }
}
```

## 📝 Testing Your Models

### Test YOLO Model

```bash
# Use webcam
python vision/yolov11_pipeline.py

# Use video file (modify config.py stream_url)
# stream_url = "path/to/test/video.mp4"
```

### Test LSTM Model

```bash
# Test with violence video
python vision/violence_lstm_inference.py --video dataset/Violence/V_1.mp4

# Test with non-violence video
python vision/violence_lstm_inference.py --video dataset/NonViolence/NV_1.mp4
```

## 🐛 Troubleshooting

### Training Issues

**Problem**: CUDA out of memory
```bash
# Solution: Reduce batch size in training scripts
```

**Problem**: Training very slow
```bash
# Check if GPU is being used
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Problem**: Model not found during inference
```bash
# Check if training completed successfully
ls training/violence_detection/yolo11n_violence/weights/
ls training/models/yolo_lstm/

# If missing, re-run training
```

### Inference Issues

**Problem**: Low accuracy on videos
```bash
# Solution: Train for more epochs or with more data
# Adjust confidence threshold in config.py
```

**Problem**: Slow inference
```bash
# For YOLO: Ensure using GPU
# For LSTM: Use --no-display flag and smaller batch size
```

## 📚 Additional Resources

- **Training Details**: See `training/README.md`
- **Model Architecture**: See `training/yolo_lstm_model.py`
- **Dataset Format**: Videos in `dataset/Violence/` and `dataset/NonViolence/`

## 🎓 What's Next?

1. ✅ Train your models on the provided dataset
2. ✅ Test models with sample videos
3. ✅ Integrate with main Vision Guard system
4. ✅ Deploy to production
5. ✅ Monitor and improve based on real-world performance

## 💡 Tips for Better Results

1. **More Data**: Add more videos to improve accuracy
2. **Balanced Dataset**: Ensure equal violence/non-violence samples
3. **Data Quality**: Remove corrupted videos
4. **Hyperparameter Tuning**: Experiment with learning rates and batch sizes
5. **Ensemble**: Use both models together for best results

---

**Questions?** Check the detailed documentation in `training/README.md` or the main project docs.

Happy Training! 🎉
