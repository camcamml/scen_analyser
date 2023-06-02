import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
##
## To get thoes information : o3d.t.io.RealSenseSensor.list_devices()

jsParams = {
                "serial": "",
                "color_format": "RS2_FORMAT_RGB8",
                "color_resolution": "0,480",
                "depth_format": "RS2_FORMAT_Z16",
                "depth_resolution": "0,480",
                "fps": "30",
                "visual_preset": ""
             }

rs = o3d.t.io.RealSenseSensor()
rs_cfg = o3d.t.io.RealSenseSensorConfig(jsParams)
rs.init_sensor(rs_cfg, 0)
rs.start_capture(True)  # true: start recording with capture

#o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault

spinholeCamera = o3d.io.read_pinhole_camera_intrinsic('camera_primesense.json')


for fid in range(150):
    im_rgbd = rs.capture_frame(True, True)  # wait for frames and align them
    # process im_rgbd.depth and im_rgbd.color
    print(type(im_rgbd))
    # open3d.cuda.pybind.t.geometry.PointCloud


    #pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(im_rgbd, pinholeCamera)

    #intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    #print(intrinsic.intrinsic_matrix)

    intrinsic = o3d.core.Tensor(np.array([  [525.  ,  0. , 319.5],
                                            [  0.  ,525. , 239.5],
                                            [  0.  ,  0. ,   1. ]]))

    pcd = o3d.t.geometry.PointCloud.create_from_rgbd_image(im_rgbd, intrinsic)
    print(type(pcd))

    # Create an empty point cloud
    # Use pcd.point to access the points' attributes

    print("-----")

    pcd2 = pcd.to_legacy()
    o3d.visualization.draw_geometries([pcd2])


    # print(pcd.point["positions"])
    # print(pcd.point["normals"])
    # # print(pcd.point["colors"])

    # pcd = pcd.PointCloud()

    # print(np.asarray(pcd.points))

    # plt.subplot(1, 2, 1)
    # plt.title('SUN grayscale image')
    # plt.imshow(im_rgbd.color)
    # plt.title('SUN depth image')
    # plt.imshow(im_rgbd.depth)
    # plt.show()
    #
    # # pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
    #     im_rgbd,
    #     o3d.camera.PinholeCameraIntrinsic(
    #         o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # o3d.visualization.draw_geometries([pcd], zoom=0.5)

rs.stop_capture()
