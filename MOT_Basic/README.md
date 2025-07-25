# MOT in Real-Time Videos using YOLOv8 + DeepSORT
## Project Information
In this project, I have developed a basic Motion Tracker in Real Time using YOLOv8 + DeepSORT.
---

##  Objective

To detect and track multiple people in a video stream and annotate them with persistent IDs, enabling consistent tracking across frames.

---

```bash
##  Folder Structure
Multiple_object_tracking/
├── input_video.mp4 # Input video
├── output_video.mp4 # Output video with tracking
└── mot_tracker.py # Main script
```
---

##  Output

The output video (`output_video.mp4`) will display:<br>
- Bounding boxes around each detected person<br>
- A unique ID per person for consistent tracking<br>

---
##  Requirements

Install the required libraries using:

```bash
pip install ultralytics opencv-python deep_sort_realtime
```
## Models Used
- YOLOv8n: For real-time object detection (person detection only)
- DeepSORT: For ID assignment and tracking across frames
## How to Run
- Place your video<br>
Put your input video (e.g. input_video.mp4) in the folder Multiple_object_tracking.<br>
- Run the script<br>
Open a terminal in the folder and run:<br>
```bash
python mot_tracker.py
```
