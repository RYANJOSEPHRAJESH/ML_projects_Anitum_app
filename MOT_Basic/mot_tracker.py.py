import os
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# === Configuration ===
folder = 'Multiple_object_tracking'
input_path = os.path.join('input_video.mp4')
output_path = os.path.join('output_video.mp4')

# Load YOLOv8 (for detection) and DeepSORT (for tracking)
detector = YOLO('yolov8n.pt')
tracker = DeepSort(max_age=30)

# Load video
cap = cv2.VideoCapture(input_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Output video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# === Main Loop ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect objects
    results = detector.predict(frame, classes=[0], verbose=False)  # Only person class
    boxes = results[0].boxes

    detections = []
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        conf = float(box.conf[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

    # Track
    tracks = tracker.update_tracks(detections, frame=frame)

    # Draw boxes
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)

# === Cleanup ===
cap.release()
out.release()
print(f"Output saved to: {output_path}")
