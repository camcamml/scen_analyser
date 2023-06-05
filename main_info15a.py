import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from ultralytics import YOLO
import torchvision
import sys, os
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def conv_xyxy_to_cxcywh(image, xyxy):
    center_x = ((xyxy[0] + xyxy[2]) / 2) / image.shape[1]
    center_y = ((xyxy[1] + xyxy[3]) / 2) / image.shape[0]
    w = (xyxy[2] - xyxy[0]) / image.shape[1]
    h = (xyxy[3] - xyxy[1]) / image.shape[0]
    return [center_x, center_y, w, h]

def estimate_w_h(object_image, object_distance, scale_distance=123., scale_size_px=52., scale_size_cm=10.):
    w = (object_distance / scale_distance) * (scale_size_cm / scale_size_px) * object_image.shape[0]
    h = (object_distance / scale_distance) * (scale_size_cm / scale_size_px) * object_image.shape[1]
    return round(w, 2), round(h, 2)

def cut_box_cv2_image(image, box):
    box = box.astype(int)
    cropped_image = image[box[0]:box[2], box[1]:box[3]]
    return cropped_image

try:
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

    model = YOLO('yolov8n.pt')

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
        
        results = model(color_image, conf=0.5, show=False)
         
        objects = []
        for result in results:
            for object in result.boxes.cpu():
                classe = object.cls.numpy()[0]
                prob = object.conf.numpy()[0]
                box_xyxy = object.xyxy.numpy()[0]
                objects.append({'name':result.names[classe], 
                                'classe':int(classe), 
                                'proba':prob, 
                                'box_xyxy':box_xyxy, 
                                'box_cxcywh':conv_xyxy_to_cxcywh(color_image, box_xyxy)
                                })
        

        if len(objects) != 0:
            start_point = (objects[0]['box_xyxy'][0].astype(int), objects[0]['box_xyxy'][1].astype(int))
            end_point = (objects[0]['box_xyxy'][2].astype(int), objects[0]['box_xyxy'][3].astype(int))
            center = ((objects[0]['box_cxcywh'][0]*color_image.shape[1]).astype(int), (objects[0]['box_cxcywh'][1]*color_image.shape[0]).astype(int))
            #print(center)
            dist_obj_cm = round(depth_frame.get_distance(center[0], center[1])*100, 2)
            cuted_object = cut_box_cv2_image(color_image, objects[0]['box_xyxy'])
            print(cuted_object.shape)
            obj_w_h = estimate_w_h(cuted_object, dist_obj_cm)
            print(obj_w_h)

            color_image_with_box = cv2.rectangle(color_image, start_point, end_point, (255,255,0), 2) 
            color_image_with_circle = cv2.circle(color_image_with_box, center, 5, (0,0,255), 2)
            dimension_string = '' + str(obj_w_h[0]) + '*' + str(obj_w_h[1]) + 'cm, width*height'
            printed_string = objects[0]['name'] + ', ' + str(dist_obj_cm) + " cm"
            cv2.putText(color_image_with_circle, printed_string, (start_point[0], end_point[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            cv2.putText(color_image_with_circle, dimension_string, (start_point[0], end_point[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        # au prof jsp à quoi ça sert
        #coverage = [0] * 64
        #for y in range(height):
        #    for x in range(width):
        #        dist = depth_frame.get_distance(x, y)
        #        if 0 < dist and dist < 1:
        #           coverage[x // 10] += 1
                    

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image_with_circle, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                             interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))


        cv2.namedWindow('RealSence', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSence', images)
        cv2.waitKey(1)


    exit(0)
# except rs.error as e:
#    # Method calls agaisnt librealsense objects may throw exceptions of type pylibrs.error
#    print("pylibrs.error was thrown when calling %s(%s):\n", % (e.get_failed_function(), e.get_failed_args()))
#    print("    %s\n", e.what())
#    exit(1)
except Exception as e:
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, "Line : " + str(exc_tb.tb_lineno))
    print(e)
    pass
