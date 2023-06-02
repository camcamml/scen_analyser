import cv2
from ultralytics import YOLO


# Load a pretrained YOLO model (recommended for training)
model = YOLO('yolov5n.pt')

# Train the model using the 'coco128.yaml' dataset for 3 epochs
#results = model.train(data='coco128.yaml', epochs=3)

# Evaluate the model's performance on the validation set
#results = model.val()


image_test = cv2.imread("image_test/bus.jpg")
cv2.imshow(image_test)
results = model(image_test, show=True, boxes=True)

input("Press Enter to continue...")

