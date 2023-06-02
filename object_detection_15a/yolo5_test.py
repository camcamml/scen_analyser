import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
model = YOLO('yolov8n.pt')
img = cv2.imread("image_test/rgb_52_621.jpg")  # or file, Path, PIL, OpenCV, numpy, list

## Inference
results = model(img, show=True)

def conv_xyxy_to_cxcywh(image, xyxy):
    center_x = ((xyxy[0] + xyxy[2])/2)/image.shape[1]
    center_y = ((xyxy[1] + xyxy[3])/2)/image.shape[0]
    w = (xyxy[2]-xyxy[0])/image.shape[1]
    h = (xyxy[3]-xyxy[1])/image.shape[0]
    return [center_x, center_y, w, h]
        
objects = []
for result in results:
    for object in result.boxes.cpu():
        classe = object.cls.numpy()[0]
        prob = object.conf.numpy()[0]
        box_xyxy = object.xyxy.numpy()[0]
        objects.append((result.names[classe], prob, box_xyxy))

print(objects[0][2])
print(img.shape)

cxcywh = conv_xyxy_to_cxcywh(img, objects[0][2])
print(cxcywh)
print(" ".join(str(elem) for elem in cxcywh))

input('press key to continue')