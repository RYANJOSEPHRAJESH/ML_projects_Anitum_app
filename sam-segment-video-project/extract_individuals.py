import cv2
import os
import numpy as np
from ultralytics import YOLO, SAM

# === Config ===
video_path = "input_video.mp4"
output_dir = "output_videos"
os.makedirs(output_dir, exist_ok=True)

# === Load Models ===
tracker = YOLO("yolov8n.pt")  # for detection + tracking
segmenter = SAM("sam2.1_b.pt")  # for segmentation

# === Read Video ===
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === Storage for each person ID ===
person_buffers = {}  # id: [frame1, frame2, ...]

frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    print(f"Processing frame {frame_num}")

    # Run tracking
    results = tracker.track(source=frame, persist=True, classes=[0], verbose=False)
    detections = results[0].boxes

    if detections.id is None:
        continue

    ids = detections.id.cpu().numpy().astype(int)
    bboxes = detections.xyxy.cpu().numpy().astype(int)

    # Segment each tracked person
    masks = segmenter(frame, bboxes=bboxes)

    for i, result in enumerate(masks):
        person_id = ids[i]
        mask = result.masks.data[0].cpu().numpy().astype(np.uint8)  # Binary mask

        if mask.shape != frame.shape[:2]:
            continue

        # Apply mask
        person_only = cv2.bitwise_and(frame, frame, mask=mask)

        # Optional: remove background (make black)
        background_removed = person_only

        # Store in buffer
        if person_id not in person_buffers:
            person_buffers[person_id] = []
        person_buffers[person_id].append(background_removed)

cap.release()

# === Write videos ===
for person_id, frames in person_buffers.items():
    save_path = os.path.join(output_dir, f"person_{person_id}.mp4")
    h, w, _ = frames[0].shape
    out = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for frame in frames:
        out.write(frame)
    out.release()
    print(f"Saved: {save_path}")

