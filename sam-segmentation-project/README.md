# Person Segmentation from Images using YOLOv8 and SAM 2.1
## Project Information
In this project, I have used SAM 2.1 and YOLO v8n to extract images of people from the input image, remove background with high accuracy.
## Libraries

- torch
- ultralytics
## Neural Network

- SAM2.1
- YOLOv8
```bash
## 📁 Folder Structure
project/
│
├── xyz.jpg # Input image (your image with people)
├── person_1.jpg # Output cropped image (Person 1)
├── person_2.jpg # Output cropped image (Person 2)
├── person_3.jpg # Output cropped image (Person 3)
├── person_segment.py # Main Python script
```
## Usage
Step 1: Place your input image
Make sure your image (e.g. xyz.jpg) is in the same folder as the script.<br>
Step 2: Run the script<br>
```bash
python person_segment.py
```

## ⚙️ Requirements

Install the following dependencies:

```bash
pip install ultralytics opencv-python matplotlib torch torchvision
```
## Notes
- YOLO provides rough bounding boxes; SAM refines them using image segmentation.
- This combination ensures both detection speed and segmentation accuracy.
