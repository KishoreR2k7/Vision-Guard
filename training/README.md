# Violence Detection Training Guide

This directory contains all the scripts needed to train violence detection models using your dataset.

## 📁 Dataset Structure

Your dataset should be organized as:
```
dataset/
├── Violence/       # Violence videos (V_1.mp4, V_2.mp4, ...)
└── NonViolence/    # Non-violence videos (NV_1.mp4, NV_2.mp4, ...)
```

## 🚀 Training Pipeline

### Option 1: YOLO-only Violence Detection (Frame-level)

This approach trains YOLO to detect violence in individual frames.

**Step 1: Prepare Dataset**
```bash
cd training
python prepare_dataset.py
```
This will:
- Extract frames from videos
- Create train/val/test splits (70/15/15)
- Generate YOLO format annotations
- Create `data.yaml` configuration

**Step 2: Train YOLO Model**
```bash
python train_yolo.py
```
This will:
- Train YOLOv11-nano on violence detection
- Save best model weights
- Export to ONNX format for inference
- Generate training metrics and visualizations

**Training Configuration:**
- Model: YOLOv11-nano
- Epochs: 100
- Batch size: 16
- Image size: 640x640
- Learning rate: 0.01

**Output:**
- `violence_detection/yolo11n_violence/weights/best.pt` - Best PyTorch model
- `violence_detection/yolo11n_violence/weights/best.onnx` - ONNX model for inference
- Training curves and metrics in the project directory

---

### Option 2: YOLO + LSTM (Temporal Video-level Detection)

This approach uses LSTM to analyze temporal patterns across video frames.

**Step 1: Prepare Video Sequences**
```bash
cd training
python prepare_sequences.py
```
This will:
- Extract sequences of 16 frames from each video
- Preprocess frames (resize to 224x224, normalize)
- Create train/val/test splits
- Save as pickle files

**Step 2: Train YOLO + LSTM Model**
```bash
python train_yolo_lstm.py
```
This will:
- Train ResNet50 + LSTM model
- Use attention mechanism for temporal fusion
- Save best model based on validation accuracy
- Generate confusion matrix and training curves

**Training Configuration:**
- Sequence length: 16 frames
- Frame size: 224x224
- Hidden size: 256
- LSTM layers: 2
- Epochs: 50
- Batch size: 4
- Learning rate: 0.0001

**Output:**
- `models/yolo_lstm/best_model.pth` - Best model checkpoint
- `models/yolo_lstm/final_model.pth` - Final model
- `models/yolo_lstm/training_history.json` - Training metrics
- `models/yolo_lstm/training_curves.png` - Visualization
- `models/yolo_lstm/confusion_matrix_best.png` - Confusion matrix

---

## 🔍 Inference

### Using YOLO Model (Real-time Frame Detection)

The trained YOLO model is automatically used in the main pipeline:
```bash
cd ..
python vision/yolov11_pipeline.py
```

The pipeline will automatically detect and use the violence detection model if available.

### Using YOLO + LSTM Model (Video Classification)

For video-level violence detection:
```bash
python vision/violence_lstm_inference.py --video path/to/video.mp4 --output output.mp4
```

Options:
- `--video`: Path to input video (required)
- `--model`: Path to trained model (default: `training/models/yolo_lstm/best_model.pth`)
- `--output`: Path to save output video with annotations
- `--no-display`: Disable real-time display

Example:
```bash
# Process a violence video
python vision/violence_lstm_inference.py --video dataset/Violence/V_1.mp4 --output results/violence_output.mp4

# Process a non-violence video
python vision/violence_lstm_inference.py --video dataset/NonViolence/NV_1.mp4 --output results/nonviolence_output.mp4

# Process without display (faster)
python vision/violence_lstm_inference.py --video test_video.mp4 --no-display
```

---

## 📊 Model Comparison

| Model | Pros | Cons | Use Case |
|-------|------|------|----------|
| **YOLO-only** | - Fast inference<br>- Frame-level detection<br>- Can detect specific violent actions | - No temporal context<br>- May have false positives | Real-time monitoring, quick frame analysis |
| **YOLO + LSTM** | - Temporal understanding<br>- Better accuracy<br>- Video-level classification | - Slower inference<br>- Requires sequence of frames | Video classification, forensic analysis |

---

## 🛠️ Requirements

Make sure you have all dependencies installed:
```bash
pip install -r requirements.txt
```

Key dependencies:
- `torch` >= 2.2.0 (with CUDA support)
- `torchvision` >= 0.17.0
- `ultralytics` >= 8.1.0
- `opencv-python` >= 4.8.1
- `onnxruntime-gpu` >= 1.17.0
- `scikit-learn`
- `matplotlib`
- `seaborn`

---

## 💡 Tips for Better Results

1. **Data Augmentation**: The YOLO training includes augmentation (flips, rotation, etc.)

2. **Transfer Learning**: Both models use pretrained weights:
   - YOLO uses YOLOv11-nano pretrained on COCO
   - LSTM uses ResNet50 pretrained on ImageNet

3. **Hyperparameter Tuning**: Adjust in the training scripts:
   - Batch size (reduce if GPU memory insufficient)
   - Learning rate
   - Number of epochs
   - Sequence length (for LSTM)

4. **GPU Memory**: 
   - YOLO training: ~8GB recommended
   - LSTM training: ~12GB recommended
   - Reduce batch size if OOM errors occur

5. **Training Time**:
   - YOLO: ~2-4 hours (depends on GPU)
   - LSTM: ~1-2 hours (fewer epochs needed)

---

## 🐛 Troubleshooting

**Issue**: CUDA out of memory
- **Solution**: Reduce batch size in training scripts

**Issue**: Slow training on CPU
- **Solution**: Install CUDA and PyTorch with GPU support

**Issue**: Model not found during inference
- **Solution**: Check model paths, ensure training completed successfully

**Issue**: Poor accuracy
- **Solution**: Train for more epochs, adjust learning rate, ensure dataset quality

---

## 📈 Next Steps

1. **Evaluate Models**: Test on held-out test set
2. **Fine-tune**: Adjust hyperparameters based on results
3. **Deploy**: Integrate trained models into production pipeline
4. **Monitor**: Track model performance in real-world scenarios

---

## 📝 Notes

- The dataset contains ~1000 violence and ~1000 non-violence videos
- Training will automatically use GPU if available
- Models are saved with best validation accuracy
- All preprocessing is handled automatically

For questions or issues, check the main project documentation.
