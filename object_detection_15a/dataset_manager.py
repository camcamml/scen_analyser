import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import os

#src = '/path/to/src/dir/filename.txt'
#dest = '/path/to/dest/dir/filename.txt'
#os.replace(src, dest)

model = YOLO('yolov8n.pt')
#model = torch.hub.load('ultralytics/yolov8', 'yolov8s')  # or yolov5n - yolov5x6, custom
img = cv2.imread("image_test/rgb_52_621.jpg")  # or file, Path, PIL, OpenCV, numpy, list

results = model(img, show=True)

def conv_xyxy_to_cxcywh(image, xyxy):
    center_x = ((xyxy[0] + xyxy[2]) / 2) / image.shape[1]
    center_y = ((xyxy[1] + xyxy[3]) / 2) / image.shape[0]
    w = (xyxy[2] - xyxy[0]) / image.shape[1]
    h = (xyxy[3] - xyxy[1]) / image.shape[0]
    return [center_x, center_y, w, h]


#pandas_results = results.pandas()
#print(pandas_results.infos())

objects = []
for result in results:
    for object in result.boxes.cpu():
        classe = object.cls.numpy()[0]
        prob = object.conf.numpy()[0]
        box_xyxy = object.xyxy.numpy()[0]
        objects.append({'name':result.names[classe], 'classe':int(classe), 'proba':prob, 'box_xyxy':box_xyxy})

print(objects[0]['box_xyxy'])
print(img.shape)

cxcywh = conv_xyxy_to_cxcywh(img, objects[0]['box_xyxy'])

label_name = "rgb_52.txt"
path_to_train_label = "datasets/dataset_15a/train/labels/" + label_name
classe = objects[0]['classe']

label = "" + str(classe) + ' ' + " ".join(str(elem) for elem in cxcywh)

with open(path_to_train_label, 'w') as f:
    f.write(label)

input('press key to continue')