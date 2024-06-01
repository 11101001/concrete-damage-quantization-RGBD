import qiexiang
import pyzed.sl as sl
import glob
import numpy as np

# 类三：相机标定执行函数(双目校正)
class Stereo_calibrate():  # 执行校正
    def __init__(self):
        # 终止条件
        self.criteria = (qiexiang.TERM_CRITERIA_EPS + qiexiang.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # 准备对象点，棋盘方块交界点排列：6行8列 如 (0,0,0), (1,0,0), (2,0,0) ....,(WW6,8,0)
        self.row,self.col = 6,8
        self.objpoints = np.zeros((self.row * self.col, 3), np.float32)
        self.objpoints[:, :2] = np.mgrid[0:self.row, 0:self.col].T.reshape(-1, 2)

    def exe(self,dir_l,dir_r):
        objectpoints = [] # 真实世界中的3d点
        imgpoints_l = []
        imgpoints_r = []
        # 标定所用图像
        images_l = glob.glob('%s/*'%dir_l)
        images_r = glob.glob('%s/*' % dir_r)
        for i in range(len(images_l)):
            img_l = qiexiang.imread(images_l[i])
            gray_l = qiexiang.cvtColor(img_l, qiexiang.COLOR_BGR2GRAY)
            img_r = qiexiang.imread(images_r[i])
            gray_r = qiexiang.cvtColor(img_r, qiexiang.COLOR_BGR2GRAY)
            # 寻找到棋盘角点
            ret1, corners_l = qiexiang.findChessboardCorners(img_l, (self.row, self.col), None)
            ret2, corners_r = qiexiang.findChessboardCorners(img_r, (self.row, self.col), None)
            # 如果找到，添加对象点，图像点（细化之后）
            if ret1 == True and ret2 == True:
                # 添加每幅图的对应3D-2D坐标
                objectpoints.append(self.objpoints)
                corners_l = qiexiang.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1),self.criteria)
                imgpoints_l.append(corners_l)
                corners_r = qiexiang.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), self.criteria)
                imgpoints_r.append(corners_r)

                qiexiang.drawChessboardCorners(img_l, (self.row, self.col), corners_l, ret1)
                qiexiang.drawChessboardCorners(img_r, (self.row, self.col), corners_r, ret2)
                view  = np.concatenate((img_l, img_r), axis=1)
                qiexiang.namedWindow('View')
                qiexiang.imshow("View", qiexiang.resize(view,(1920,540)))
                qiexiang.waitKey(0)
                qiexiang.destroyAllWindows()
        # 利用单目校正函数实现相机内参初始化
        ret, m1, d1, _, _ = qiexiang.calibrateCamera(objectpoints, imgpoints_l, gray_l.shape[::-1], None, None)
        ret, m2, d2, _, _= qiexiang.calibrateCamera(objectpoints, imgpoints_l, gray_l.shape[::-1], None, None)
        # config
        flags = 0
        # flags |= cv2.CALIB_FIX_ASPECT_RATIO
        flags |= qiexiang.CALIB_USE_INTRINSIC_GUESS
        # flags |= cv2.CALIB_SAME_FOCAL_LENGTH
        # flags |= cv2.CALIB_ZERO_TANGENT_DIST
        flags |= qiexiang.CALIB_RATIONAL_MODEL
        # flags |= cv2.CALIB_FIX_K1
        # flags |= cv2.CALIB_FIX_K2
        # flags |= cv2.CALIB_FIX_K3
        # flags |= cv2.CALIB_FIX_K4
        # flags |= cv2.CALIB_FIX_K5
        # flags |= cv2.CALIB_FIX_K6
        stereocalib_criteria = (qiexiang.TERM_CRITERIA_COUNT +
                                qiexiang.TERM_CRITERIA_EPS, 100, 1e-5)
        # 输入参数：真实3d坐标点，左相机像素点、右相机像素点、左内参、左畸变、右内参、右畸变、图像尺寸、一些配置
        # 输出值：未知、左内参、左畸变、右内参、右畸变（迭代优化后的）、旋转矩阵、平移向量、本质矩阵、基础矩阵
        ret, m1, d1,m2, d2, R, t, E, F = qiexiang.stereoCalibrate(objectpoints,imgpoints_l,imgpoints_r,
                                                                       m1, d1,m2, d2, gray_l.shape[::-1],
                                                                       criteria=stereocalib_criteria, flags=flags)
        # 构建单应性矩阵
        plane_depth = 40000000.0  # arbitrary plane depth
        # TODO: Need to understand effect of plane_depth. Why does this improve some boards' cals?
        n = np.array([[0.0], [0.0], [-1.0]])
        d_inv = 1.0 / plane_depth
        H = (R - d_inv * np.dot(t, n.transpose()))
        H = np.dot(m2, np.dot(H, np.linalg.inv(m1)))
        H /= H[2, 2]
        # rectify Homography for right camera
        disparity = (m1[0, 0] * t[0] / plane_depth)
        H[0, 2] -= disparity
        H = H.astype(np.float32)
        print(ret,'\n左相机矩阵：%s\n左相机畸变:%s\n右相机矩阵：%s\n右相机畸变:%s\n旋转矩阵:%s\n平移向量:%s'
                  '\n本质矩阵E:%s\n基础矩阵F:%s\n单应性矩阵H:%s'
              %(m1, d1,m2, d2, R, t, E, F,H))


if __name__ == "__main__":
    # 双目校正
    cal = Stereo_calibrate()
    cal.exe(r'imgL',r'imgR')


