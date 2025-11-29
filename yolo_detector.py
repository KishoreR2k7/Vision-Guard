import torch
from ultralytics import YOLO
import numpy as np

class YOLODetector:
    """
    Loads a trained YOLO model and detects objects in frames.
    Triggers suspicious event flag for violence-related labels.
    """
    def __init__(self, weights_path, suspicious_labels=None, conf_threshold=0.4):
        self.model = YOLO(weights_path)
        self.conf_threshold = conf_threshold
        # Define suspicious labels (customize as needed)
        self.suspicious_labels = suspicious_labels or ["fight", "punch", "kick", "weapon", "violence"]

    def detect(self, frame):
        results = self.model(frame)
        labels, boxes, confidences = [], [], []
        suspicious = False
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < self.conf_threshold:
                    continue
                label = self.model.names[cls]
                labels.append(label)
                boxes.append(box.xyxy[0].cpu().numpy())
                confidences.append(conf)
                if label in self.suspicious_labels:
                    suspicious = True
        return labels, boxes, confidences, suspicious
