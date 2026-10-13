Real-Time Object Detection with YOLOv3 and OpenCV

A Python program that detects everyday objects (people, phones, cups, chairs and more) in a live webcam stream. It runs a pretrained YOLOv3 network through OpenCV's dnn module and draws a labeled box around each detected object.

Note: The model is pretrained on the COCO dataset (80 classes). This project does not train a new model. It shows how to load a Darknet network, run inference on video frames, and clean up the results.

Features
Live detection from the webcam
80 object classes from the COCO dataset (listed in coco.names)
Confidence filtering to remove weak detections
Non-maximum suppression (NMS) to remove duplicate boxes for the same object
Colored boxes and class labels drawn on each frame
Press q to quit
How It Works
Load the model: cv2.dnn.readNetFromDarknet loads yolov3.cfg (network structure) and yolov3.weights (pretrained weights). Class names are read from coco.names.
Prepare each frame: cv2.dnn.blobFromImage resizes the frame to 320 x 320, scales pixel values to the 0–1 range, and swaps the color channels from BGR to RGB.
Run the network: a forward pass through the three YOLO output layers returns candidate boxes with class scores.
Filter detections: candidates with confidence below 0.3 are dropped. The box center, width and height are converted to pixel coordinates.
Remove duplicates: cv2.dnn.NMSBoxes (score threshold 0.5, overlap threshold 0.4) keeps only the best box for each object.
Draw results: a colored rectangle and the class label are drawn on the frame and shown in a window.
Project Structure
object-detection/
├── main.py          # Detection code
├── coco.names       # The 80 class names
├── yolov3.cfg       # YOLOv3 network configuration
└── yolov3.weights   # Pretrained weights (not included, see below)
Requirements
Python 3.10 or newer
A webcam
Python packages:
pip install opencv-python numpy
Setup and Usage
Clone this repository.
Download the pretrained weights file yolov3.weights (about 237 MB) from the official YOLO website: https://pjreddie.com/media/files/yolov3.weights
Place yolov3.weights in the same folder as main.py. If yolov3.cfg is missing, get it from the official Darknet repository: https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
Run the program from inside the project folder, because the file paths are relative:
python main.py
Point the camera at objects. Press q in the video window to stop.
Settings You Can Change
Setting	Where	Current value	Effect
Input size	detect_objects	320 x 320	Larger (for example 416 x 416) can be more accurate but slower
Confidence threshold	get_box_dimensions	0.3	Higher means fewer, more certain detections
NMS score / overlap	draw_labels	0.5 / 0.4	Controls how duplicate boxes are removed
Camera	start_detection	VideoCapture(0)	Change the number to use another camera
Limitations
Detection speed depends on your computer. This version runs on the CPU.
Only the 80 COCO classes can be detected.
Small, blurry or partly hidden objects may be missed.
Box colors are random, so they change every time the program starts.
No accuracy or speed measurements are included yet.
Ideas for Improvement
Measure and display frames per second (FPS)
Run on GPU through OpenCV's CUDA backend
Detect objects in image and video files, not only from the webcam
Fine-tune a model on a custom dataset
Compare YOLOv3 with a lighter model such as YOLOv3-tiny
Technologies

Python, OpenCV (cv2.dnn), NumPy, YOLOv3, COCO dataset

Credits

YOLOv3 by Joseph Redmon and Ali Farhadi. Model files and the COCO class list come from the official YOLO / Darknet project.
