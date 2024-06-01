import pyzed.sl as sl
import math

# 初始化ZED相机
init_params = sl.InitParameters()
init_params.camera_resolution = sl.RESOLUTION.HD720
init_params.camera_fps = 30
zed = sl.Camera()
if not zed.is_opened():
    print("Opening ZED Camera...")
status = zed.open(init_params)
if status != sl.ERROR_CODE.SUCCESS:
    print(repr(status))
    exit()

# 创建Mat对象来存储深度图像
depth_image = sl.Mat()

# 开始捕获图像
runtime_parameters = sl.RuntimeParameters()
if zed.grab(runtime_parameters) == sl.ERROR_CODE.SUCCESS:
    # 从ZED相机中检索深度图像
    zed.retrieve_image(depth_image, sl.VIEW.DEPTH)

    # 定义一个三维点列表，用于存储深度图中的点
    points = []

    # 遍历深度图像的每个像素
    for y in range(depth_image.get_height()):
        for x in range(depth_image.get_width()):
            # 获取当前像素的深度值
            depth_value = depth_image.get_value(x, y)[0] 
            #print(depth_value)
            # 如果深度值有效（不为零）
            if depth_value > 0:
                # 将像素坐标转换为相机坐标系下的三维坐标
                point_cloud = sl.Mat()
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA)
                point = point_cloud.get_value(x, y)
                x_c = point[0] * 1000  # 深度值乘以1000，转换为毫米
                y_c = point[1] * 1000
                z_c = point[2] * 1000

                # 添加三维点到列表中
                points.append([x_c, y_c, z_c])

    # 计算体积（假设点列表定义了一个闭合的多边形）
    volume = 0.0
    for i in range(len(points)):
        j = (i + 1) % len(points)
        volume += (points[i][0] * points[j][1] - points[j][0] * points[i][1])
    volume = abs(volume) / 2.0

    # 输出体积
    print("Volume:", volume, "cubic millimeters")

# 关闭ZED相机
zed.close()