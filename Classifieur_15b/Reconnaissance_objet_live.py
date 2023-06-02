import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import torch
import torchvision
import time
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

try:
    # load yolo5 model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom

    # Create a context object. This object owns the handles to all connected realsense devices
    pipeline = rs.pipeline()
    config = rs.config()

    # Get device product line for setting a supporting resolution
    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    pipeline_profile = config.resolve(pipeline_wrapper)
    device = pipeline_profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    print("Config ---- ")
    intr = pipeline_profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    extr = pipeline_profile.get_stream(rs.stream.color).as_video_stream_profile()
    print(intr)
    print(intr)
    print("Config ---- ")
    found_rgb = False
    for s in device.sensors:
        if s.get_info(rs.camera_info.name) == 'RGB Camera':
            found_rgb = True
            break
    if not found_rgb:
        print("The demo requires Depth camera with Color sensor")
        exit(0)

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    if device_product_line == 'L500':
        config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    #print(config)


    while True:
        # This call waits until a new coherent set of frames is available on a device
        # Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(color_frame.profile)
        #print(depth_intrin)
        #print(color_intrin)
        #print(depth_to_color_extrin)

        if not depth_frame or not color_frame:
            continue

        width = depth_frame.get_width()
        height = depth_frame.get_height()
        dist = depth_frame.get_distance(int(width / 2), int(height / 2))

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                             interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))

        results = model(images)

        # Results
        result = results.pandas()  # or .show(), .save(), .crop(), .pandas(), etc.
        data = results.pandas().xyxy[0]
        #print(data.info)

        lst = []
        for index, row in data.iterrows():
            line = []
            line.append(row["name"])
            line.append(int(row["xmin"]))
            line.append(int(row["ymin"]))
            line.append(int(row["xmax"]))
            line.append(int(row["ymax"]))
            lst.append(line)

        #print(lst)

        for l in lst:
            cv2.rectangle(images,(l[1],l[2]),(l[3],l[4]),(0,255,0),1)
            cv2.putText(images, l[0], (l[1],l[2]), 0, 0.3, (0, 255, 0))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

        # # bounding box in (xmin, ymin, xmax, ymax) format
        # # top-left point=(xmin, ymin), bottom-right point = (xmax, ymax)
        # bbox = [290, 115, 405, 385]
        # bbox = torch.tensor(bbox, dtype=torch.int)
        # print(bbox)
        # print(bbox.size())
        # bbox = bbox.unsqueeze(0)
        # print(bbox.size())
        # # draw bounding box on the input image
        # img=draw_bounding_boxes(images, bbox, width=3, colors=(255,255,0))

        # # transform it to PIL image and display
        # img = torchvision.transforms.ToPILImage()(img)
        # img.show()

    exit(0)
# except rs.error as e:
#    # Method calls agaisnt librealsense objects may throw exceptions of type pylibrs.error
#    print("pylibrs.error was thrown when calling %s(%s):\n", % (e.get_failed_function(), e.get_failed_args()))
#    print("    %s\n", e.what())
#    exit(1)
except Exception as e:
    print(e)
    pass


# read input image
# img = read_image('cat.jpg')

# # bounding box in (xmin, ymin, xmax, ymax) format
# # top-left point=(xmin, ymin), bottom-right point = (xmax, ymax)
# bbox = [290, 115, 405, 385]
# bbox = torch.tensor(bbox, dtype=torch.int)
# print(bbox)
# print(bbox.size())
# bbox = bbox.unsqueeze(0)
# print(bbox.size())
# # draw bounding box on the input image
# img=draw_bounding_boxes(img, bbox, width=3, colors=(255,255,0))

# # transform it to PIL image and display
# img = torchvision.transforms.ToPILImage()(img)
# img.show()