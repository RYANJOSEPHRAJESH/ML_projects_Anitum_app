import os
import cv2
import numpy as np
from ultralytics import YOLO, SAM

# === Step 1: Setup ===
image_path = 'xyz.jpg'
folder_path = os.path.dirname(image_path) or '.'

# Load models
detector = YOLO('yolov8n.pt')
segmenter = SAM('sam2.1_b.pt')

# === Step 2: Detect People ===
detection_results = detector(image_path)
boxes = detection_results[0].boxes

# Extract bounding boxes of 'person' class
bboxes = []
for det in boxes:
    if int(det.cls) == 0:
        x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
        bboxes.append([x1, y1, x2, y2])

# === Step 3: Segment with SAM ===
segmentation_results = segmenter(image_path, bboxes=bboxes)
image = cv2.imread(image_path)
person_count = 1

# === Step 4: Mask + Crop each person ===
for result in segmentation_results:
    if hasattr(result, 'masks') and result.masks is not None:
        for mask in result.masks.data:
            mask = mask.cpu().numpy().astype(np.uint8) * 255
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))  # Resize to match image

            # Apply mask to image
            masked_image = cv2.bitwise_and(image, image, mask=mask)

            # Optional: crop to non-zero area of mask
            y_indices, x_indices = np.where(mask > 0)
            if len(x_indices) == 0 or len(y_indices) == 0:
                continue  # Skip empty masks

            x1, x2 = np.min(x_indices), np.max(x_indices)
            y1, y2 = np.min(y_indices), np.max(y_indices)
            cropped = masked_image[y1:y2, x1:x2]

            # Save masked image
            save_path = os.path.join(folder_path, f"person_{person_count}.png")
            cv2.imwrite(save_path, cropped)
            print(f"Saved: {save_path}")
            person_count += 1


