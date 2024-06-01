import pyzed.sl as sl
import cv2
import os
import open3d as o3d
import numpy as np
import time
import math
import numpy as np
import sys
def save_point_cloud(point_cloud, file_path):
    np.savetxt(file_path, point_cloud, delimiter=' ', fmt='%f')
#def solve():
def save_point_cloud(zed, filename) :
    print("Saving Point Cloud...")
    tmp = sl.Mat()
    zed.retrieve_measure(tmp, sl.MEASURE.XYZRGBA)
    saved = (tmp.write("./pointcloud/"+filename + ".xyz") == sl.ERROR_CODE.SUCCESS) # 如果想换成其他格式，替换这个地方就可以
    #print(str(saved))
    if saved :
        print("Done")
    else :
        print("Failed... Please check that you have permissions to write on disk")
    return str("./pointcloud/"+filename + ".xyz") 

def main():
    # Create a Camera object
    zed = sl.Camera()
    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720  # Use HD1080 video mode
    init_params.camera_fps = 30  # Set fps at 30
    # Open the camera
    err = zed.open(init_params)
    if err != sl.ERROR_CODE.SUCCESS:
        exit(1)
    
    calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
    # Focal length of the left eye in pixels焦距
    focal_left_x = calibration_params.left_cam.fx
    focal_left_y = calibration_params.left_cam.fy
    # First radial distortion coefficient畸变系数
    k1 = calibration_params.left_cam.disto[0]
    # Translation between left and right eye on z-axis
    
    h_fov = calibration_params.left_cam.h_fov
    # Capture 50 frames and stop
    print(focal_left_x,focal_left_y)
    print(k1)
    print(h_fov)

    i = 0
    image = sl.Mat()
    disparity = sl.Mat()  # 视差值
    disparity = sl.Mat()  # 视差值
    dep = sl.Mat()  # 深度图
    depth = sl.Mat()  # 深度值
    point_cloud = sl.Mat()  # 点云数据
    
    mirror_ref = sl.Transform()
    mirror_ref.set_translation(sl.Translation(2.75,4.0,0))
    tr_np = mirror_ref.m

     # 获取分辨率
    resolution = zed.get_camera_information().camera_configuration.resolution 
    w, h = resolution.width , resolution.height 
    x,y = int(w/2),int(h/2)  # 中心点 
    print(x,y) 
    runtime_parameters = sl.RuntimeParameters()  
    runtime_parameters.enable_fill_mode	 = True

    while True:
        # 获取最新的图像，修正它们，并基于提供的RuntimeParameters(深度，点云，跟踪等)计算测量值。
        if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:  # 相机成功获取图象
            # 获取图像
            timestamp = zed.get_timestamp(sl.TIME_REFERENCE.CURRENT)  # 获取图像被捕获时的时间点
            zed.retrieve_image(image, sl.VIEW.LEFT)  # image：容器，sl.VIEW.LEFT：内容
            img = image.get_data()  # 转换成图像数组，便于后续的显示或者储存
            # 获取视差值
            zed.retrieve_measure(disparity,sl.MEASURE.DISPARITY,sl.MEM.CPU)
            dis_map = disparity.get_data()
            # 获取深度
            zed.retrieve_measure(depth,sl.MEASURE.DEPTH,sl.MEM.CPU)  # 深度值  
            zed.retrieve_image(dep,sl.VIEW.DEPTH)  # 深度图                     
            depth_map = depth.get_data() 
            dep_map = dep.get_data() 
            # 获取点云
            #print(depth_map)

            zed.retrieve_measure(point_cloud,sl.MEASURE.XYZBGRA,sl.MEM.CPU) 
            point_cloud_np = point_cloud.get_data()
            point_cloud_np.dot(tr_np)

            #print('时间点',timestamp.get_seconds(),'中心点视差值',dis_map[x,y],'中心点深度值',depth_map[x,y],'中心点云数据',point_map[x,y])

            # 利用cv2.imshow显示视图，并对想要的视图进行保存
            # view1  =  cv2.resize(img,(512,512))
            #view2  =  cv2.resize(depth_map,(512,512))
            cv2.imshow("View1", img)        
            cv2.imshow("View2", dep_map)  
            key = cv2.waitKey(1) 
            if key & 0xFF == 27:  # esc退出
                break
            if key & 0xFF == ord('s'):  # 图像保存
                date_string = "V{:0>3d}".format(i)
                saved = save_point_cloud(zed, date_string)
                #print(saved)
                pcd = o3d.io.read_point_cloud(saved)  # 这里的cat.ply替换成需要查看的点云文件
                o3d.visualization.draw_geometries([pcd]) 
                savePath1 = os.path.join("./images", "V{:0>3d}.png".format(i))  # 注意根目录是否存在"./images"文件夹
                savePath2 = os.path.join("./dimages", "V{:0>3d}.png".format(i)) 
                cv2.imwrite(savePath1, img) 
                cv2.imwrite(savePath2, dep_map) 
                np.savetxt("./depthvalue/V{:0>3d}depth_values.txt".format(i), depth_map)
                
                #solve(img,depth_map,point_cloud) 
                
            i = i + 1 
    zed.close() 


 
if __name__ == "__main__":
    main()