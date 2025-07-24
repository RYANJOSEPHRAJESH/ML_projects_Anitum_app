# Person-wise Segmentation Background Removal from Video
This project processes a video, detects individual people in each frame, removes their backgrounds, and saves each person's movement across the video as a separate video file.<br>

It uses:<br>

- YOLOv8 for real-time person detection.

- SAM (Segment Anything Model) for precise person segmentation.

- OpenCV for frame handling and video writing.

## Libraries

- torch and torchvision
- ultralytics
- opencv-python
- matplotlib

## Models Used:

  - YOLOv8n
  - SAM (Segment Anything Model)
  - cv2.VideoWriter (not a model)
```bash
## Folder Structure
project-folder/
│
├── input.mp4                # Input video
├── extract_individuals.py   # Main script
├── person_clips/            # Output folder containing cropped videos
```
## Requirements
Install the following dependencies:<br>
```bash
pip install ultralytics opencv-python torch torchvision matplotlib
```
## How to Run
Step 1: Place your input video<br>
Make sure your video (e.g., input.mp4) is in the same folder as the script.<br>
Step 2: Run the script<br>
```bash
python extract_individuals.py
```
Step 3: Output<br>
The script will:<br>

Detect people in each frame.<br>

Segment each one with SAM.<br>

Track and save each individual into a separate video file inside person_clips/.<br>
## Notes
Ideal for videos with 3–5 people moving with minimal occlusion.<br>

Each person’s movement is captured and exported without background.<br>


