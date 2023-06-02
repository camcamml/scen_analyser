import pyrealsense2 as rs
import numpy as np
import cv2
import torch
from ultralytics import YOLO

#model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
model = YOLO('yolov5s.pt')
img = 'https://ultralytics.com/images/zidane.jpg'  # or file, Path, PIL, OpenCV, numpy, list

print(type(img))

## Inference
results = model(img, show=True)


def conv_xyxy_to_cxcywh(image, xyxy):
    centerX = int(((xyxy[0] + xyxy[2])/2)/image.shape[0])
    centerY = int(((xyxy[1] + xyxy[3])/2)/image.shape[1])
    w = int(abs(xyxy[2]-centerX)/image.shape[0])
    h = int(abs(xyxy[3]-centerY)/image.shape[1])
    return [centerX, centerY, w, h]
        
objects = []
for result in results:
    for object in result.boxes.cpu():
        classe = object.cls.numpy()[0]
        prob = object.conf.numpy()[0]
        box_xyxy = object.xyxy.numpy()[0]
        objects.append((result.names[classe], prob, box_xyxy))

print(objects[0][2])
img = np.array(img)
print(img.shape)

cxcywh = conv_xyxy_to_cxcywh(img, objects[0][2]) 
print(cxcywh)