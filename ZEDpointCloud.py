import qiexiang
import numpy as np
import time
import random
import math
import pyzed.sl as sl

# -----------------------------------双目相机的基本参数---------------------------------------------------------
#   left_camera_matrix          左相机的内参矩阵
#   right_camera_matrix         右相机的内参矩阵
#
#   left_distortion             左相机的畸变系数    格式(K1,K2,P1,P2,0)
#   right_distortion            右相机的畸变系数
# -------------------------------------------------------------------------------------------------------------
# 左镜头的内参，如焦距
left_camera_matrix = np.array([[922.74805888,0,861.07449695],[0,842.38453769,738.75205406],[0,0,1]])
right_camera_matrix = np.array([[511.8428182, 1.295112628, 317.310253], [0, 513.0748795, 269.5885026], [0., 0., 1.]])

# 畸变系数,K1、K2、K3为径向畸变,P1、P2为切向畸变
left_distortion = np.array([[-0.046645194, 0.077595167, 0.012476819, -0.000711358, 0]])
right_distortion = np.array([[-0.061588946, 0.122384376, 0.011081232, -0.000750439, 0]])

# 旋转矩阵
R = np.array([[0.999911333, -0.004351508, 0.012585312],
              [0.004184066, 0.999902792, 0.013300386],
              [-0.012641965, -0.013246549, 0.999832341]])
# 平移矩阵
T = np.array([-120.3559901, -0.188953775, -0.662073075])

size = (640, 480)

# 畸变矫正与立体矫正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = qiexiang.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)

# 校正查找映射表,将原始图像和校正后的图像上的点一一对应起来
left_map1, left_map2 = qiexiang.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, qiexiang.CV_16SC2)
right_map1, right_map2 = qiexiang.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, qiexiang.CV_16SC2)
print(Q)
WIN_NAME = 'Deep disp'
qiexiang.namedWindow(WIN_NAME, qiexiang.WINDOW_AUTOSIZE)


# --------------------------鼠标回调函数---------------------------------------------------------
#   event               鼠标事件
#   param               输入参数
# -----------------------------------------------------------------------------------------------
def onmouse_pick_points(event, x, y, flags, param):
    if event == qiexiang.EVENT_LBUTTONDOWN:
        threeD = param
        print('\n像素坐标 x = %d, y = %d' % (x, y))
        # print("世界坐标是：", threeD[y][x][0], threeD[y][x][1], threeD[y][x][2], "mm")
        print("世界坐标xyz 是：", threeD[y][x][0] / 1000.0, threeD[y][x][1] / 1000.0, threeD[y][x][2] / 1000.0, "m")

        distance = math.sqrt(threeD[y][x][0] ** 2 + threeD[y][x][1] ** 2 + threeD[y][x][2] ** 2)
        distance = distance / 1000.0  # mm -> m
        print("距离是：", distance, "mm")


# 读取视频
fps = 0.0
cam = sl.Camera()
init = sl.InitParameters()
init.camera_resolution = sl.RESOLUTION.HD720

status = cam.open(init)
if status != sl.ERROR_CODE.SUCCESS:
    print(repr(sl))
    cam.close()
    exit(1)

runtime = sl.RuntimeParameters()
image_left = sl.Mat()
image_right = sl.Mat()

while True:
    cam.grab(runtime)
    # 开始计时
    t1 = time.time()
    # 左右两张图片
    cam.retrieve_image(image_left, sl.VIEW.LEFT)
    cam.retrieve_image(image_right, sl.VIEW.RIGHT)
    frame1 = image_left.get_data()
    frame2 = image_right.get_data()

    # 将BGR格式转换成灰度图片，用于畸变矫正
    imgL = qiexiang.cvtColor(frame1, qiexiang.COLOR_BGR2GRAY)
    imgR = qiexiang.cvtColor(frame2, qiexiang.COLOR_BGR2GRAY)

    # 重映射，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程。
    # 依据MATLAB测量数据重建无畸变图片,输入图片要求为灰度图
    img1_rectified = qiexiang.remap(imgL, left_map1, left_map2, qiexiang.INTER_LINEAR)
    img2_rectified = qiexiang.remap(imgR, right_map1, right_map2, qiexiang.INTER_LINEAR)

    # 转换为opencv的BGR格式
    imageL = qiexiang.cvtColor(img1_rectified, qiexiang.COLOR_GRAY2BGR)
    imageR = qiexiang.cvtColor(img2_rectified, qiexiang.COLOR_GRAY2BGR)

    # ------------------------------------SGBM算法----------------------------------------------------------
    #   blockSize                   深度图成块，blocksize越低，其深度图就越零碎，0<blockSize<10
    #   img_channels                BGR图像的颜色通道，img_channels=3，不可更改
    #   numDisparities              SGBM感知的范围，越大生成的精度越好，速度越慢，需要被16整除，如numDisparities
    #                               取16、32、48、64等
    #   mode                        sgbm算法选择模式，以速度由快到慢为：STEREO_SGBM_MODE_SGBM_3WAY、
    #                               STEREO_SGBM_MODE_HH4、STEREO_SGBM_MODE_SGBM、STEREO_SGBM_MODE_HH4。精度反之
    # ------------------------------------------------------------------------------------------------------
    blockSize = 3
    img_channels = 3
    stereo = qiexiang.StereoSGBM_create(minDisparity=1,
                                   numDisparities=64,
                                   blockSize=blockSize,
                                   P1=8 * img_channels * blockSize * blockSize,
                                   P2=32 * img_channels * blockSize * blockSize,
                                   disp12MaxDiff=-1,
                                   preFilterCap=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=100,
                                   speckleRange=100,
                                   mode=qiexiang.STEREO_SGBM_MODE_HH) 
    # 计算视差
    disparity = stereo.compute(img1_rectified, img2_rectified)

    # 归一化函数算法，生成深度图（灰度图）
    disp = qiexiang.normalize(disparity, disparity, alpha=0, beta=255, norm_type=qiexiang.NORM_MINMAX, dtype=qiexiang.CV_8U)


    # 生成深度图（颜色图）
    dis_color = disparity
    dis_color = qiexiang.normalize(dis_color, None, alpha=0, beta=255, norm_type=qiexiang.NORM_MINMAX, dtype=qiexiang.CV_8U)
    dis_color = qiexiang.applyColorMap(dis_color, 2)

    # 计算三维坐标数据值
    threeD = qiexiang.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
    # 计算出的threeD，需要乘以16，才等于现实中的距离
    threeD = threeD * 16

    # 鼠标回调事件
    qiexiang.namedWindow("depth", qiexiang.WINDOW_AUTOSIZE)
    qiexiang.setMouseCallback("depth", onmouse_pick_points, threeD)

    qiexiang.imshow("depth", dis_color)
    qiexiang.imshow("left", frame1)
    qiexiang.imshow(WIN_NAME, disp)  # 显示深度图的双目画面
    # 若键盘按下q则退出播放
    if qiexiang.waitKey(1) & 0xff == ord('q'):
        break

# 关闭所有窗口
qiexiang.destroyAllWindows()

