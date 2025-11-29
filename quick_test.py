"""
Simple Video Test Script with Bounding Box Detection
Quick test for any trained YOLO model on test.mp4
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime

def test_video_with_model(model_path, video_path='test.mp4', conf_threshold=0.5):
    """
    Test a video with a trained YOLO model and draw bounding boxes
    
    Args:
        model_path: Path to trained .pt model file
        video_path: Path to video file (default: test.mp4)
        conf_threshold: Confidence threshold for detections (default: 0.5)
    """
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("\n💡 Available trained models:")
        print("   - runs/train/accident_detection/weights/best.pt")
        print("   - runs/train/garbage_detection/weights/best.pt")
        print("   - runs/train/garbage_detection3/weights/best.pt")
        print("   - runs/train/violence_detection/weights/best.pt")
        return
    
    # Check if video exists and resolve absolute path
    video_path = str(Path(video_path).resolve())
    if not Path(video_path).exists():
        print(f"❌ Video not found: {video_path}")
        print(f"   Current directory: {Path.cwd()}")
        return
    
    print("="*70)
    print("🎬 VIDEO DETECTION TEST")
    print("="*70)
    print(f"📦 Model: {model_path}")
    print(f"🎥 Video: {video_path}")
    print(f"   Video size: {Path(video_path).stat().st_size / 1024 / 1024:.2f} MB")
    print(f"🎯 Confidence: {conf_threshold}")
    print("="*70)
    
    # Load model
    print("\n⏳ Loading model...")
    model = YOLO(model_path)
    print("✅ Model loaded successfully!")
    print(f"   Model classes: {model.names}")
    print(f"   Number of classes: {len(model.names)}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"\n📹 Video Info:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Frames: {total_frames}")
    print(f"   Duration: {total_frames/fps:.1f}s")
    
    # Setup output video with clear naming based on input
    input_name = Path(video_path).stem  # Get filename without extension
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = Path(model_path).parent.parent.name  # Get model folder name
    output_path = f"output_{input_name}_{model_name}_{timestamp}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Verify VideoWriter opened successfully
    if not out.isOpened():
        print("❌ Error: Could not open VideoWriter")
        cap.release()
        return
    
    print(f"\n💾 Output will be saved as: {output_path}")
    print(f"   Processing input: {Path(video_path).name}")
    print(f"   Video properties: {int(total_frames)} frames, {fps:.2f} fps, {width}x{height}")
    print(f"\n🚀 Processing... Press 'Q' to stop\n")
    
    frame_count = 0
    total_detections = 0
    failed_reads = 0
    max_failed_reads = 5  # Allow some failed reads before giving up
    
    while True:
        ret, frame = cap.read()
        if not ret:
            failed_reads += 1
            if failed_reads >= max_failed_reads:
                print(f"\n⚠️  Stopped after {failed_reads} consecutive failed frame reads")
                break
            # Try to continue - sometimes OpenCV has temporary read issues
            continue
        
        # Reset failed reads counter on successful read
        failed_reads = 0
        
        frame_count += 1
        
        # Run detection
        results = model(frame, conf=conf_threshold, verbose=False)
        
        # Draw bounding boxes
        frame_detections = 0
        for result in results:
            boxes = result.boxes
            for box in boxes:
                frame_detections += 1
                
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # Get class name
                class_name = model.names[cls] if cls in model.names else f"class_{cls}"
                
                # Choose color based on class
                if class_name.lower() == 'violence':
                    color = (0, 0, 255)  # Red
                elif class_name.lower() == 'nonviolence':
                    color = (0, 255, 0)  # Green
                elif class_name.lower() == 'accident':
                    color = (0, 0, 255)  # Red
                elif class_name.lower() == 'garbage3':
                    color = (0, 165, 255)  # Orange
                else:
                    color = (255, 0, 0)  # Blue
                
                # Draw bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
                
                # Draw label with background
                label = f"{class_name}: {conf:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(frame, (int(x1), int(y1) - label_h - 15), 
                            (int(x1) + label_w + 10, int(y1)), color, -1)
                cv2.putText(frame, label, (int(x1) + 5, int(y1) - 8), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        total_detections += frame_detections
        
        # Add frame info overlay
        info_bg = frame.copy()
        cv2.rectangle(info_bg, (0, 0), (width, 50), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, info_bg, 0.3, 0)
        
        info = f"Frame: {frame_count}/{total_frames} | Detections: {frame_detections} | Total: {total_detections}"
        cv2.putText(frame, info, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Add progress bar
        progress = frame_count / total_frames
        bar_width = int(width * progress)
        cv2.rectangle(frame, (0, height - 15), (bar_width, height), (0, 255, 0), -1)
        
        # Show percentage
        percent_text = f"{progress*100:.1f}%"
        cv2.putText(frame, percent_text, (width - 100, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Write frame - verify it succeeded
        if not out.write(frame):
            print(f"\n⚠️  Warning: Failed to write frame {frame_count}")
        
        # Display
        cv2.imshow('Detection Test - Press Q to quit', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n⏸️  Stopped by user")
            break
        
        # Progress update
        if frame_count % (fps * 5) == 0:
            print(f"   ⏳ {frame_count}/{int(total_frames)} frames ({progress*100:.1f}%) | Detections: {total_detections}")
    
    # Final progress update
    print(f"\n✅ Completed: {frame_count}/{int(total_frames)} frames processed")
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Check if we processed the full video
    completion_rate = (frame_count / total_frames) * 100 if total_frames > 0 else 0
    if completion_rate < 95:
        print(f"\n⚠️  Warning: Only processed {completion_rate:.1f}% of the video")
        print(f"   Expected: {int(total_frames)} frames, Got: {frame_count} frames")
    
    # Summary
    print("\n" + "="*70)
    print("📊 RESULTS")
    print("="*70)
    print(f"   Frames Processed: {frame_count}/{total_frames}")
    print(f"   Total Detections: {total_detections}")
    if frame_count > 0:
        print(f"   Average Detections/Frame: {total_detections/frame_count:.2f}")
    print(f"\n✅ Output saved: {output_path}")
    print("="*70)


def main():
    """Main function with interactive model selection"""
    
    print("\n🎯 QUICK VIDEO TEST WITH BOUNDING BOXES")
    print("="*70)
    
    # Check for trained models
    models = {
        '1': ('Accident Detection', 'runs/train/accident_detection/weights/best.pt'),
        '2': ('Garbage Detection', 'runs/train/garbage_detection2/weights/best.pt'),
        '3': ('Garbage Detection v3', 'runs/train/garbage_detection3/weights/best.pt'),
        '4': ('Violence Detection', 'runs/train/violence_detection/weights/best.pt'),
    }
    
    print("\n📦 Available Models:")
    available_models = {}
    for key, (name, path) in models.items():
        if Path(path).exists():
            print(f"   [{key}] {name} ✅")
            available_models[key] = (name, path)
        else:
            print(f"   [{key}] {name} ❌ (not trained)")
    
    if not available_models:
        print("\n❌ No trained models found!")
        print("\n💡 Train a model first:")
        print("   python train_all_models.py")
        return
    
    # Get user choice
    print("\n" + "="*70)
    choice = input("Select model (1-4) or press Enter for default: ").strip()
    
    if not choice and '3' in available_models:
        choice = '3'  # Default to garbage_detection3
    elif not choice and available_models:
        choice = list(available_models.keys())[0]
    
    if choice not in available_models:
        print(f"❌ Invalid choice: {choice}")
        return
    
    model_name, model_path = available_models[choice]
    
    # Get video path
    video_path = input("\nVideo path (press Enter for test.mp4): ").strip()
    
    # Remove quotes if user pasted path with quotes
    video_path = video_path.strip('"').strip("'")
    
    if not video_path:
        video_path = 'test.mp4'
    
    # Resolve to absolute path
    video_path = str(Path(video_path).resolve())
    print(f"\n📍 Using video: {video_path}")
    
    # Get confidence threshold
    conf_input = input("\nConfidence threshold (press Enter for 0.5): ").strip()
    conf_threshold = float(conf_input) if conf_input else 0.5
    
    # Run test
    print(f"\n🎬 Testing {model_name} on {video_path}...\n")
    test_video_with_model(model_path, video_path, conf_threshold)


if __name__ == "__main__":
    # Quick test mode if you want to skip interactive menu
    # Uncomment and modify this to run directly:
    # test_video_with_model('runs/train/garbage_detection3/weights/best.pt', 'test.mp4', 0.5)
    
    # Or run interactive mode:
    main()
