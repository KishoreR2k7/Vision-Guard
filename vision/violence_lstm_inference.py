"""
Inference Script for YOLO + LSTM Violence Detection
This script uses the trained YOLO + LSTM model to detect violence in videos
"""

import torch
import cv2
import numpy as np
from pathlib import Path
import sys
import argparse
from collections import deque

# Add training directory to path
sys.path.append(str(Path(__file__).parent.parent / "training"))
from yolo_lstm_model import SimplifiedYOLOLSTM


class ViolenceLSTMDetector:
    """Violence detector using YOLO + LSTM model"""
    def __init__(self, model_path, sequence_length=16, frame_size=(224, 224), device='cuda'):
        self.sequence_length = sequence_length
        self.frame_size = frame_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model
        print(f"Loading model from {model_path}...")
        self.model = SimplifiedYOLOLSTM(num_classes=2, hidden_size=256, num_layers=2).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"✅ Model loaded successfully on {self.device}")
        
        # Frame buffer
        self.frame_buffer = deque(maxlen=sequence_length)
        
        # Class names
        self.classes = ['NonViolence', 'Violence']
    
    def preprocess_frame(self, frame):
        """Preprocess a single frame"""
        # Resize
        frame = cv2.resize(frame, self.frame_size)
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Normalize
        frame = frame.astype(np.float32) / 255.0
        return frame
    
    def predict(self, frames):
        """
        Predict violence for a sequence of frames
        Args:
            frames: List or array of frames (seq_len, H, W, 3)
        Returns:
            class_idx: Predicted class index
            confidence: Prediction confidence
        """
        # Convert to tensor
        frames_tensor = torch.from_numpy(np.array(frames))  # (seq_len, H, W, 3)
        frames_tensor = frames_tensor.permute(0, 3, 1, 2)  # (seq_len, 3, H, W)
        frames_tensor = frames_tensor.unsqueeze(0).to(self.device)  # (1, seq_len, 3, H, W)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(frames_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        class_idx = predicted.item()
        confidence_val = confidence.item()
        
        return class_idx, confidence_val
    
    def process_video(self, video_path, output_path=None, display=True):
        """
        Process a video file and detect violence
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            display: Whether to display results in real-time
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n📹 Processing video: {Path(video_path).name}")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Total frames: {total_frames}")
        
        # Video writer
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        frame_count = 0
        violence_detected_count = 0
        predictions = []
        
        print("\n🔄 Processing frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess and add to buffer
            processed_frame = self.preprocess_frame(frame)
            self.frame_buffer.append(processed_frame)
            
            # Make prediction when buffer is full
            class_name = "Buffering..."
            confidence = 0.0
            
            if len(self.frame_buffer) == self.sequence_length:
                class_idx, confidence = self.predict(list(self.frame_buffer))
                class_name = self.classes[class_idx]
                
                predictions.append({
                    'frame': frame_count,
                    'class': class_name,
                    'confidence': confidence
                })
                
                if class_idx == 1:  # Violence
                    violence_detected_count += 1
            
            # Draw results on frame
            color = (0, 0, 255) if class_name == "Violence" else (0, 255, 0)
            
            # Background for text
            cv2.rectangle(frame, (10, 10), (400, 100), (0, 0, 0), -1)
            
            # Text
            cv2.putText(frame, f"Status: {class_name}", (20, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Confidence: {confidence:.2%}", (20, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Frame counter
            cv2.putText(frame, f"Frame: {frame_count}/{total_frames}", (20, height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Write frame
            if writer:
                writer.write(frame)
            
            # Display
            if display:
                cv2.imshow('Violence Detection - YOLO + LSTM', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n⚠️ Stopped by user")
                    break
            
            frame_count += 1
            
            # Progress
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"   Progress: {progress:.1f}% ({frame_count}/{total_frames})")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 Detection Summary")
        print("=" * 60)
        print(f"Total frames processed: {frame_count}")
        print(f"Violence detected in: {violence_detected_count} sequences")
        
        violence_percentage = (violence_detected_count / max(1, len(predictions))) * 100
        print(f"Violence percentage: {violence_percentage:.2f}%")
        
        if violence_percentage > 50:
            print("🚨 Result: VIOLENCE DETECTED")
        else:
            print("✅ Result: NO VIOLENCE")
        
        if output_path:
            print(f"\n💾 Output saved to: {output_path}")
        
        return predictions


def main():
    parser = argparse.ArgumentParser(description='Violence Detection using YOLO + LSTM')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--model', type=str, default='./training/models/yolo_lstm/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--output', type=str, default=None, help='Path to save output video')
    parser.add_argument('--no-display', action='store_true', help='Disable video display')
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Error: Model not found at {model_path}")
        print("   Please train the model first using train_yolo_lstm.py")
        return
    
    # Check if video exists
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"❌ Error: Video not found at {video_path}")
        return
    
    # Initialize detector
    detector = ViolenceLSTMDetector(
        model_path=str(model_path),
        sequence_length=16,
        frame_size=(224, 224)
    )
    
    # Process video
    detector.process_video(
        video_path=str(video_path),
        output_path=args.output,
        display=not args.no_display
    )


if __name__ == "__main__":
    main()
