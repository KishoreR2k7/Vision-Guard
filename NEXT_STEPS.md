# Quick Action Guide - Train Remaining Models

## Current Status:
- ✅ **Accident Detection** - COMPLETE (99.5% mAP)
- ⏸️ **Garbage Detection** - Need to fix numpy
- ❌ **Violence Detection** - Dataset empty

---

## 🔧 Step 1: Fix NumPy Issue

### Option A (Recommended):
```powershell
pip install numpy==1.26.4
```

### Option B (If using conda):
```powershell
conda install numpy=1.26 -c conda-forge
```

### Verify Fix:
```powershell
python -c "import numpy; print(numpy.__version__); import numpy._core; print('OK')"
```

---

## 🗑️ Step 2: Train Garbage Detection

### After fixing numpy, run:
```powershell
python train_garbage_only.py
```

### Or use the interactive plan:
```powershell
python training_plan.py
```

**Expected Results:**
- Training time: ~10-15 minutes
- Similar performance to accident detection
- Model saved to: `runs/train/garbage_detection/weights/best.pt`

---

## 🥊 Step 3: Violence Detection Dataset

You have **3 options**:

### Option A: Use Prepared Dataset (if available)
```powershell
# Check if you have prepared dataset
Get-ChildItem dataset_prepared\images\train | Measure-Object

# If images exist, train with:
python -c "from ultralytics import YOLO; YOLO('yolo11n.pt').train(data='dataset_prepared/data.yaml', epochs=100, batch=16, imgsz=640, name='violence_detection')"
```

### Option B: Download Violence Dataset
```powershell
# Download from one of these sources:
# 1. Kaggle: Real Life Violence Dataset
# 2. Roboflow: Violence Detection datasets
# 3. UCF Crime Dataset

# After downloading, organize as:
# dataset/violence/
#   ├── images/
#   │   ├── train/
#   │   ├── val/
#   │   └── test/
#   └── labels/ (same structure)
```

### Option C: Use Your Own Videos
1. Place videos in `dataset/violence/videos/`
2. Extract frames and create labels
3. Train model

---

## 🚀 Quick Training Command

### Train All Remaining (after fixes):
```powershell
# Fix numpy first
pip install numpy==1.26.4

# Then train
python train_new_datasets.py
```

### Train One at a Time:
```powershell
# Garbage only
python train_garbage_only.py

# Violence (when dataset ready)
python training/train_yolo.py
```

---

## 📊 Check Training Status

### View trained models:
```powershell
Get-ChildItem runs\train -Directory
```

### Check model performance:
```powershell
# View results
code runs\train\accident_detection\results.png
code runs\train\garbage_detection\results.png

# Test model
python -c "from ultralytics import YOLO; YOLO('runs/train/accident_detection/weights/best.pt').predict('test.mp4', save=True)"
```

---

## 🎯 Recommended Order:

1. ✅ **DONE** - Accident Detection trained
2. **NEXT** - Fix numpy → Train Garbage
3. **THEN** - Prepare violence dataset → Train Violence

---

## 💡 Pro Tips:

### For Garbage Detection:
- Should train successfully after numpy fix
- Expected accuracy: 90-95%
- Fast training: ~10 minutes

### For Violence Detection:
- Needs video dataset (200+ videos recommended)
- Or use image frames with labels
- More complex than garbage/accident
- Consider temporal analysis (LSTM) for better results

### Monitor Training:
```powershell
# Watch GPU usage
nvidia-smi -l 1

# View logs
Get-Content runs\train\<model_name>\results.csv
```

---

## 🆘 Troubleshooting:

### If numpy error persists:
```powershell
pip uninstall numpy
pip install numpy==1.26.4 --no-cache-dir
```

### If CUDA out of memory:
- Reduce batch size: `batch=8`
- Reduce image size: `imgsz=480`

### If dataset not found:
- Check paths in data.yaml
- Verify images and labels exist
- Use absolute paths

---

## 📞 Quick Commands Reference:

```powershell
# Check everything
python training_plan.py

# Train garbage (after numpy fix)
python train_garbage_only.py

# Check GPU
nvidia-smi

# View training results
code runs\train\garbage_detection\results.png

# Test model
python -c "from ultralytics import YOLO; model = YOLO('runs/train/garbage_detection/weights/best.pt'); model.predict('test.mp4')"
```

---

**Last Updated:** November 27, 2025
