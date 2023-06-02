import cv2
import numpy as np
#import cvlib as cv
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov5s.pt')


