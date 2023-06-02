import cv2
import numpy as np
#import cvlib as cv
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov8n.pt')

image_test_1 = cv2.imread("image_test/bus.jpg")
image_test_2 = cv2.imread("image_test/rgb_52_621.jpg")
images_test = [image_test_1, image_test_2]
results = model(images_test, conf=0.5, show=True)
plt.imshow(image_test_2)

objects = []
for result in results:
    for object in result.boxes.cpu():
        classe = object.cls.numpy()[0]
        prob = object.conf.numpy()[0]
        box = object.xyxy.numpy()[0]
        objects.append((result.names[classe], prob, box))

print(objects)

input("Press Enter to continue...")
