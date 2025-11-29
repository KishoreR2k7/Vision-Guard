import time
from camera_stream import CameraStream
from yolo_detector import YOLODetector
from temporal_buffer import FrameBuffer
from temporal_model_tsm import TSMClassifier

# ====== CONFIGURATION ======
CAMERA_SRC = 0  # 0 for webcam, or use IP camera URL
YOLO_WEIGHTS = 'training/violence_detection/yolo11n_violence_optimized/weights/best.pt'  # Path to your YOLO weights
TSM_MODEL = 'models/tsm_model.pt'
FRAME_SKIP = 3  # Run YOLO every Nth frame
BUFFER_SIZE = 40

# ===========================

def main():
    # Initialize modules
    cam = CameraStream(CAMERA_SRC)
    yolo = YOLODetector(YOLO_WEIGHTS)
    buffer = FrameBuffer(max_size=BUFFER_SIZE)
    tsm = TSMClassifier(TSM_MODEL)

    frame_count = 0
    print("[INFO] Starting real-time violence detection pipeline...")
    try:
        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue
            buffer.push(frame)
            frame_count += 1

            # Run YOLO every Nth frame
            if frame_count % FRAME_SKIP == 0:
                labels, boxes, confidences, suspicious = yolo.detect(frame)
                if suspicious:
                    print(f"[YOLO] Suspicious event detected: {labels}")
                    clip = buffer.get_clip()
                    if len(clip) == BUFFER_SIZE:
                        result = tsm.classify_clip(clip)
                        print(f"[TSM] Temporal classification: {result}")
            # Optional: display frame (for debugging)
            # cv2.imshow('Frame', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
    except KeyboardInterrupt:
        print("[INFO] Stopping...")
    finally:
        cam.stop()
        # cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
