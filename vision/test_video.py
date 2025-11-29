"""
Quick Video Tester for Violence Detection
Test the pipeline on a video file before training
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import settings
    YOLO_MODEL = settings.yolo_model_path
    CONFIDENCE = settings.confidence_threshold
except:
    YOLO_MODEL = "yolo11n.onnx"
    CONFIDENCE = 0.5

# COCO classes for testing
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class SimpleYOLO:
    def __init__(self, model_path, confidence_threshold=0.5):
        print(f"Loading model: {model_path}")
        self.session = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        self.confidence_threshold = confidence_threshold

    def preprocess(self, frame):
        resized = cv2.resize(frame, (640, 640))
        normalized = resized / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(transposed, axis=0).astype(np.float32)

    def postprocess(self, outputs):
        detections = []
        for output in outputs[0][0]:
            confidence = output[4]
            if confidence > self.confidence_threshold:
                class_scores = output[5:]
                class_id = np.argmax(class_scores)
                class_confidence = class_scores[class_id]
                
                if class_confidence > self.confidence_threshold:
                    detections.append({
                        'class_id': int(class_id),
                        'confidence': float(confidence * class_confidence),
                        'bbox': output[:4].tolist()
                    })
        return detections

    def infer(self, frame):
        input_tensor = self.preprocess(frame)
        outputs = self.session.run(None, {self.input_name: input_tensor})
        return self.postprocess(outputs)


def test_video(video_path, output_path=None):
    """Test detection on a video file"""
    
    # Check video exists
    if not Path(video_path).exists():
        print(f"❌ Error: Video not found: {video_path}")
        return
    
    # Load model
    model_path = Path(YOLO_MODEL)
    if not model_path.exists():
        print(f"❌ Error: Model not found: {model_path}")
        print("   Please download yolo11n.onnx or train the violence model first")
        return
    
    print("=" * 60)
    print("🎬 Video Detection Test")
    print("=" * 60)
    
    model = SimpleYOLO(str(model_path), confidence_threshold=CONFIDENCE)
    
    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 Video: {Path(video_path).name}")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.1f} seconds")
    
    # Video writer
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"   Output: {output_path}")
    
    print("\n🔄 Processing video... (Press 'q' to quit)")
    print("-" * 60)
    
    frame_count = 0
    detection_count = 0
    class_counts = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        detections = model.infer(frame)
        
        if detections:
            detection_count += 1
        
        # Draw detections
        for det in detections:
            bbox = det['bbox']
            x, y, w, h = bbox
            frame_h, frame_w = frame.shape[:2]
            
            # Convert YOLO format to pixel coordinates
            x1 = int((x - w/2) * frame_w / 640)
            y1 = int((y - h/2) * frame_h / 640)
            x2 = int((x + w/2) * frame_w / 640)
            y2 = int((y + h/2) * frame_h / 640)
            
            class_id = det['class_id']
            confidence = det['confidence']
            class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else f"class_{class_id}"
            
            # Track class counts
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Draw
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Info overlay
        cv2.rectangle(frame, (5, 5), (350, 80), (0, 0, 0), -1)
        cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Detections: {len(detections)}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        progress = (frame_count / total_frames) * 100
        cv2.putText(frame, f"Progress: {progress:.1f}%", (10, 75), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Write frame
        if writer:
            writer.write(frame)
        
        # Display
        cv2.imshow('Video Detection Test', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n⚠️ Stopped by user")
            break
        
        frame_count += 1
        
        # Progress update
        if frame_count % 30 == 0:
            print(f"   Processed {frame_count}/{total_frames} frames ({progress:.1f}%)")
    
    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Detection Summary")
    print("=" * 60)
    print(f"Total frames: {frame_count}")
    print(f"Frames with detections: {detection_count}")
    print(f"Detection rate: {(detection_count/max(1, frame_count))*100:.1f}%")
    
    if class_counts:
        print(f"\n🎯 Detected Objects:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   {class_name}: {count}")
    
    if output_path:
        print(f"\n💾 Output saved: {output_path}")
    
    print("\n✅ Test complete!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Test video detection')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--output', type=str, default=None, help='Path to save output video')
    
    args = parser.parse_args()
    
    test_video(args.video, args.output)


if __name__ == "__main__":
    # Quick test mode - you can modify these paths
    import sys
    
    if len(sys.argv) > 1:
        main()
    else:
        # Default video path - CHANGE THIS to test quickly
        default_video = "NV_7.mp4"
        
        print("\n" + "=" * 60)
        print("🎬 Quick Video Detection Tester")
        print("=" * 60)
        print("\nUsage:")
        print("  python test_video.py --video path/to/video.mp4")
        print("  python test_video.py --video path/to/video.mp4 --output output.mp4")
        print("\nExamples:")
        print("  python test_video.py --video ../dataset/Violence/V_1.mp4")
        print("  python test_video.py --video ../dataset/NonViolence/NV_1.mp4 --output test_result.mp4")
        print("\nOr run without arguments to test default video:")
        print(f"  Default: {default_video}")
        print("\nThis will use the current YOLO model to detect objects in the video.")
        print("After training, the violence detection model will be used automatically.")
        
        # Ask user if they want to test with default video
        response = input("\n⚡ Test with default video? (y/n): ")
        if response.lower() == 'y':
            test_video(default_video)
