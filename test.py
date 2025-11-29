"""
Test Script for Vision Guard AI Models
Tests trained models on sample video (test.mp4)
Supports: Accident Detection, Garbage Detection, Violence Detection
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse
from datetime import datetime

# Model configurations
MODELS = {
    'accident': {
        'path': 'runs/train/accident_detection/weights/best.pt',
        'name': 'Accident Detection',
        'classes': ['Accident'],
        'color': (0, 0, 255)  # Red
    },
    'garbage': {
        'path': 'runs/train/garbage_detection3/weights/best.pt',
        'name': 'Garbage Detection',
        'classes': ['garbage'],
        'color': (0, 165, 255)  # Orange
    },
    'violence': {
        'path': 'runs/train/violence_detection/weights/best.pt',
        'name': 'Violence Detection',
        'classes': ['NonViolence', 'Violence'],
        'color': (0, 0, 255)  # Red for violence
    }
}

def load_model(model_type):
    """Load trained model"""
    if model_type not in MODELS:
        print(f"❌ Unknown model type: {model_type}")
        print(f"   Available: {', '.join(MODELS.keys())}")
        return None
    
    config = MODELS[model_type]
    model_path = Path(config['path'])
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        print(f"   Please train the model first:")
        print(f"   python train_all_models.py")
        return None
    
    print(f"✅ Loading {config['name']} model...")
    print(f"   Path: {model_path}")
    
    try:
        model = YOLO(str(model_path))
        print(f"   Classes: {config['classes']}")
        return model, config
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def process_video(video_path, model, config, save_output=True, show_display=True, confidence_threshold=0.5):
    """Process video and detect objects"""
    
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 Video Information:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Total Frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.2f} seconds")
    
    # Setup output video writer
    output_writer = None
    if save_output:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"output_{config['name'].lower().replace(' ', '_')}_{timestamp}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"   Output: {output_path}")
    
    print(f"\n🔍 Processing video...")
    print(f"   Confidence threshold: {confidence_threshold}")
    print(f"   Model: {config['name']}")
    print("\n" + "="*70)
    
    frame_count = 0
    detection_count = 0
    detections_per_frame = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Run inference
            results = model(frame, conf=confidence_threshold, verbose=False)
            
            # Process results
            frame_detections = 0
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    frame_detections += 1
                    detection_count += 1
                    
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Get class name
                    class_name = config['classes'][class_id] if class_id < len(config['classes']) else f"class_{class_id}"
                    
                    # Choose color
                    if config['name'] == 'Violence Detection':
                        color = (0, 0, 255) if class_id == 1 else (0, 255, 0)  # Red for violence, green for non-violence
                    else:
                        color = config['color']
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Draw label with background
                    label = f"{class_name}: {confidence:.2f}"
                    (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (int(x1), int(y1) - label_h - 10), 
                                (int(x1) + label_w, int(y1)), color, -1)
                    cv2.putText(frame, label, (int(x1), int(y1) - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            detections_per_frame.append(frame_detections)
            
            # Add info overlay
            info_text = f"{config['name']} | Frame: {frame_count}/{total_frames} | Detections: {frame_detections}"
            cv2.rectangle(frame, (0, 0), (width, 40), (0, 0, 0), -1)
            cv2.putText(frame, info_text, (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Add progress bar
            progress = frame_count / total_frames
            bar_width = int(width * progress)
            cv2.rectangle(frame, (0, height - 10), (bar_width, height), (0, 255, 0), -1)
            
            # Save frame
            if output_writer:
                output_writer.write(frame)
            
            # Display frame
            if show_display:
                cv2.imshow(f'{config["name"]} - Test Video', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⏸️  Stopped by user")
                    break
            
            # Print progress
            if frame_count % (fps * 5) == 0:  # Every 5 seconds
                print(f"   Progress: {frame_count}/{total_frames} frames ({progress*100:.1f}%) | Detections: {detection_count}")
    
    except KeyboardInterrupt:
        print("\n\n⏸️  Interrupted by user")
    
    finally:
        cap.release()
        if output_writer:
            output_writer.release()
        if show_display:
            cv2.destroyAllWindows()
    
    # Print summary
    print("\n" + "="*70)
    print("📊 DETECTION SUMMARY")
    print("="*70)
    print(f"   Total Frames Processed: {frame_count}")
    print(f"   Total Detections: {detection_count}")
    print(f"   Average Detections/Frame: {detection_count/frame_count:.2f}")
    print(f"   Frames with Detections: {sum(1 for d in detections_per_frame if d > 0)}")
    print(f"   Detection Rate: {sum(1 for d in detections_per_frame if d > 0)/frame_count*100:.1f}%")
    
    if save_output:
        print(f"\n✅ Output saved: {output_path}")
    
    print("="*70)

def list_available_models():
    """List available trained models"""
    print("\n📦 Available Trained Models:")
    print("="*70)
    
    for model_type, config in MODELS.items():
        model_path = Path(config['path'])
        status = "✅ Ready" if model_path.exists() else "❌ Not trained"
        print(f"   {model_type:12} | {config['name']:20} | {status}")
        if model_path.exists():
            print(f"                | Path: {model_path}")
    
    print("="*70)

def main():
    parser = argparse.ArgumentParser(
        description="Test Vision Guard AI models on video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test accident detection
  python test.py --model accident --video test.mp4
  
  # Test violence detection with custom confidence
  python test.py --model violence --video test.mp4 --conf 0.7
  
  # Test without saving output
  python test.py --model garbage --video test.mp4 --no-save
  
  # List available models
  python test.py --list
        """
    )
    
    parser.add_argument('--model', type=str, choices=['accident', 'garbage', 'violence'],
                       help='Model type to use')
    parser.add_argument('--video', type=str, default='test.mp4',
                       help='Path to video file (default: test.mp4)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='Confidence threshold (default: 0.5)')
    parser.add_argument('--no-save', action='store_true',
                       help='Don\'t save output video')
    parser.add_argument('--no-display', action='store_true',
                       help='Don\'t display video window')
    parser.add_argument('--list', action='store_true',
                       help='List available models and exit')
    
    args = parser.parse_args()
    
    print("="*70)
    print("🎯 VISION GUARD AI - MODEL TEST SCRIPT")
    print("="*70)
    
    # List models and exit
    if args.list:
        list_available_models()
        return
    
    # Check if model specified
    if not args.model:
        print("\n❌ Please specify a model with --model")
        print("\nAvailable models: accident, garbage, violence")
        print("\nExample: python test.py --model violence --video test.mp4")
        print("\nOr use --list to see all available models")
        return
    
    # Load model
    result = load_model(args.model)
    if not result:
        return
    
    model, config = result
    
    # Process video
    process_video(
        video_path=args.video,
        model=model,
        config=config,
        save_output=not args.no_save,
        show_display=not args.no_display,
        confidence_threshold=args.conf
    )
    
    print("\n✅ Testing complete!")

if __name__ == "__main__":
    main()
