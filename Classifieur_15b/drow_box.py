# Import the required libraries
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes

# read input image
img = read_image('cat.jpg')

# bounding box in (xmin, ymin, xmax, ymax) format
# top-left point=(xmin, ymin), bottom-right point = (xmax, ymax)
bbox = [290, 115, 405, 385]
bbox = torch.tensor(bbox, dtype=torch.int)
print(bbox)
print(bbox.size())
bbox = bbox.unsqueeze(0)
print(bbox.size())

# draw bounding box on the input image
img=draw_bounding_boxes(img, bbox, width=3, colors=(255,255,0))

# transform it to PIL image and display
img = torchvision.transforms.ToPILImage()(img)
img.show()