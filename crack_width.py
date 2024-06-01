import cv2
import math
import random
import numpy as np
from numpy.ma import cos, sin
import matplotlib.pyplot as plt

fx = 922.74805888  # 焦距
fy = 842.38453769
center_x, center_y = 360, 640  # 图像中心点

depth_image_path = 'E:\workspace_new\ZED_DEPTH\depthvalue\V924depth_values.txt'
depth_map = np.loadtxt(depth_image_path) 
def calclen(a,b,c,d):
    #print(a,b,c,d)
    depth1 = depth_map[a, b]
    depth2 = depth_map[c, d]
    
    if(math.isnan(depth1) or math.isnan(depth2) or math.isinf(depth1) or math.isinf(depth2)):
        #print("wtf?")
        return 0
    p1 = [(a - center_x) * depth1 / fx, (b - center_y) * depth1 / fy, depth1] 
    p2 = [(c - center_x) * depth2 / fx, (d - center_y) * depth2 / fy, depth2] 
    ret = math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]) + (p1[2] - p2[2]) * (p1[2] - p2[2]))
    print(p1,p2)
    '''''
    #print(ret)
    if(math.isnan(ret)):
        print(p1)
        print(p2)
        print(depth1)
    '''
    return math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]) + (p1[2] - p2[2]) * (p1[2] - p2[2]))

def max_circle(img_path):
    '''
    计算轮廓内切圆算法
    Args:
        img_path: 输入图片路径，图片需为二值化图像
    Returns:
    '''
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 图片二值化，缺少这一步也可以，但是图像的二值化可以使图像中数据量大为减少，从而能凸显出目标的轮廓，减小计算量
    
    cv2.imshow('image',img_gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    ret, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)
    print(ret)
    cv2.imshow('image',thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # 寻找二值图像的轮廓
    contous, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    expansion_circle_list = []   # 所有裂缝最大内切圆半径和圆心列表
    # 可能一张图片中存在多条裂缝，对每一条裂缝进行循环计算0
    for c in contous:
        # 定义能包含此裂缝的最小矩形，矩形为水平方向
        left_x = min(c[:, 0, 0])
        right_x = max(c[:, 0, 0])
        down_y = max(c[:, 0, 1])
        up_y = min(c[:, 0, 1])
        # 最小矩形中最小的边除2，裂缝内切圆的半径最大不超过此距离
        upper_r = min(right_x - left_x, down_y - up_y) / 2
        # 相切二分精度precision
        precision = math.sqrt((right_x - left_x) ** 2 + (down_y - up_y) ** 2) / (2 ** 13)
        # 包含轮廓的矩形的所有像素点
        Nx = 2 ** 8
        Ny = 2 ** 8
        pixel_X = np.linspace(left_x, right_x, Nx)
        pixel_Y = np.linspace(up_y, down_y, Ny)
        # 从坐标向量中生成网格点坐标矩阵    
        xx, yy = np.meshgrid(pixel_X, pixel_Y)
        # 筛选出轮廓内所有像素点            
        in_list = []
        for i in range(pixel_X.shape[0]): 
            for j in range(pixel_X.shape[0]): 
                # cv2.pointPolygonTest可查找图像中的点与轮廓之间的最短距离.当点在轮廓外时返回负值，当点在内部时返回正值，如果点在轮廓上则返回零
                # 统计裂缝内的所有点的坐标
                if cv2.pointPolygonTest(c, (xx[i][j], yy[i][j]), False) > 0: 
                    in_list.append((xx[i][j], yy[i][j])) 
        
        in_point = np.array(in_list)
        # 随机搜索百分之一的像素点提高内切圆半径下限
        N = len(in_point)
        rand_index = random.sample(range(N), N // 100)
        rand_index.sort()
        radius = 0
        big_r = upper_r   # 裂缝内切圆的半径最大不超过此距离
        center = None
        for id in rand_index:
            tr = iterated_optimal_incircle_radius_get(c, in_point[id][0], in_point[id][1], radius, big_r, precision)
            if tr > radius:
                radius = tr
                center = (in_point[id][0], in_point[id][1])  # 只有半径变大才允许位置变更，否则保持之前位置不变
        # 循环搜索剩余像素对应内切圆半径
        loops_index = [i for i in range(N) if i not in rand_index] 
        for id in loops_index:
            tr = iterated_optimal_incircle_radius_get(c, in_point[id][0], in_point[id][1], radius, big_r, precision) 
            if tr > radius: 
                radius = tr 
                center = (in_point[id][0], in_point[id][1])  # 只有半径变大才允许位置变更，否则保持之前位置不变 
        if(center != None and radius > 0):
            expansion_circle_list.append([radius, center])       # 保存每条裂缝最大内切圆的半径和圆心 
        # 输出裂缝的最大宽度
        print('裂缝宽度：', round(radius * 2, 2))

    print('---------------')
    expansion_circle_radius_list = [i[0] for i in expansion_circle_list]   # 每条裂缝最大内切圆半径列表
    max_radius = max(expansion_circle_radius_list)
    max_center = expansion_circle_list[expansion_circle_radius_list.index(max_radius)][1]
    print('最大像素宽度：', round(max_radius * 2, 2))

    # 绘制轮廓
    cv2.drawContours(img_original, contous, -1, (0, 0, 255), -1) 
    # 绘制裂缝轮廓最大内切圆
    kuan = 0    
    for expansion_circle in expansion_circle_list:
        radius_s = expansion_circle[0]
        center_s = expansion_circle[1]

        if radius_s == max_radius:   # 最大内切圆，用蓝色标注
            cv2.circle(img_original, (int(max_center[0]), int(max_center[1])), int(max_radius), (255, 0, 0), 2) 
            #######求3d坐标下距离#######
            L = np.linspace(0, 2 * math.pi, 360)  # 确定圆散点剖分数360, 720
            pixelx = max_center[0]
            pixely = max_center[1]
            circle_X = pixelx + radius_s * cos(L)
            circle_Y = pixely + radius_s * sin(L)
            for i in range(len(circle_Y)): 
                kuan = max(kuan,calclen(int(pixelx),int(pixely),int(circle_X[i]),int(circle_Y[i])))
            kuan = kuan * 2
            print(kuan)
        else:   # 其他内切圆，用青色标注

            cv2.circle(img_original, (int(center_s[0]), int(center_s[1])), int(radius_s), (255, 245, 0), 2)     

    cv2.imshow('Inscribed_circle', img_original) 
    cv2.imwrite('inference/output/Inscribed_circle.png', img_original) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # matplotlib效果测试
    # plot_x = np.linspace(0, 2 * math.pi, 100)
    # circle_X = max_center[0] + max_radius * cos(plot_x)
    # circle_Y = max_center[1] + max_radius * sin(plot_x)
    # plt.figure()
    # plt.imshow(img_gray)
    # plt.plot(circle_X, circle_Y)
    # plt.show()


def iterated_optimal_incircle_radius_get(contous, pixelx, pixely, small_r, big_r, precision):
    '''
    计算轮廓内最大内切圆的半径
    Args:
        contous: 轮廓像素点array数组
        pixelx: 圆心x像素坐标
        pixely: 圆心y像素坐标
        small_r: 之前所有计算所求得的内切圆的最大半径，作为下次计算时的最小半径输入，只有半径变大时才允许位置变更，否则保持之前位置不变
        big_r: 圆的半径最大不超过此距离
        precision: 相切二分精度，采用二分法寻找最大半径

    Returns: 轮廓内切圆的半径
    '''
    radius = small_r
    L = np.linspace(0, 2 * math.pi, 360)  # 确定圆散点剖分数360, 720
    circle_X = pixelx + radius * cos(L)
    circle_Y = pixely + radius * sin(L)
    for i in range(len(circle_Y)):
        if cv2.pointPolygonTest(contous, (circle_X[i], circle_Y[i]), False) < 0:  # 如果圆散集有在轮廓之外的点
            return 0
    
    while big_r - small_r >= precision:  # 二分法寻找最大半径
        half_r = (small_r + big_r) / 2
        circle_X = pixelx + half_r * cos(L) 
        circle_Y = pixely + half_r * sin(L) 
        if_out = False 
        for i in range(len(circle_Y)): 
            if cv2.pointPolygonTest(contous, (circle_X[i], circle_Y[i]), False) < 0:  # 如果圆散集有在轮廓之外的点
                big_r = half_r         
                if_out = True
        if not if_out:
            small_r = half_r
    radius = small_r
    return radius


if __name__ == '__main__':
    img_gray = 'E:\workspace_new\ZED_DEPTH\inference\V924(2).png'
    img_original = cv2.imread('E:\workspace_new\ZED_DEPTH\images\V924.png')
    max_circle(img_gray)

