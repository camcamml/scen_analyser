import pyrealsense2 as rs
import numpy as np
import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
#img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list
img = 'chat.webp'  # or file, Path, PIL, OpenCV, numpy, list
print("model_loaded")
## Inference
results = model(img)

# Results
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.
results.show()