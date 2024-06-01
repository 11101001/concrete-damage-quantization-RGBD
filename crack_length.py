import cv2
import numpy as np
from skimage.filters import threshold_otsu,median
from skimage.morphology import skeletonize,dilation,disk
from skimage import io, morphology
import math
import matplotlib.pyplot as plt


# 读取RGB图像
rgb_image_path = 'E:\workspace_new\ZED_DEPTH\images\V924.png'
rgb_image = cv2.imread(rgb_image_path)

# 读取深度图像
depth_image_path = 'E:\workspace_new\ZED_DEPTH\depthvalue\V924depth_values.txt'
depth_map = np.loadtxt(depth_image_path)
#depth_image = cv2.imread(depth_image_path,  cv2.IMREAD_GRAYSCALE)

#读取分割后的图像
seg_image_path = 'E:\workspace_new\ZED_DEPTH\inference\V924(2).png'
seg_image = cv2.imread(seg_image_path)
mm = depth_map.shape[0]
nn = depth_map.shape[1]

skeleton = np.zeros((mm, nn))

# 定义相机内参
fx = 922.74805888  # 焦距
fy = 842.38453769
center_x, center_y = rgb_image.shape[1] // 2, rgb_image.shape[0] // 2  # 图像中心点


'''
###裂缝骨架处理  
def judge()      
gujia = np.zeros((mm, nn))
tmp_gujia = np.zeros((mm,nn))

# 使用BFS移除边缘像素
def judge(a,b):
    return a>=0 and a<mm and b>=0 and b<nn and gujia[a, b] != [0, 0, 0] 

def bfs(tp,lw):
    height = depth_map.shape[0],width = depth_map.shape[1]
    queue = deque([tp,lw])                             
    visited = np.zeros((height, width), dtype = np.uint8)  
    flag = True
    while queue:
        y, x = queue.popleft()
        if not (0 <= y < height and 0 <= x < width) or visited[y, x]: 
            continue 
        visited[y, x] = 1 
        tot = 0
        fg = 0
        if not np.all(image[y, x] == [0, 0, 0]):  # 非黑色像素
            if(image[y, x] != [0, 0, 0]) 
            if(judge(y - 1,x)): 
                tot += 1        
                queue.extend([(y - 1,x)]) 
            if(judge(y + 1,x)): 
                tot += 1        
                queue.extend([(y + 1,x)]) 
            if(judge(y,x - 1): 
                tot += 1
                queue.extend([(y,x - 1)]) 
            if(judge(y,x + 1)): 
                tot += 1
                queue.extend([(y,x + 1)]) 
        if(tot != 4 and (y,x) != tp and (y,x) != lw):
            if(judge(y - 1,x - 1)):
                tot += 1
            if(judge(y - 1,x + 1)):
                tot += 1
            if(judge(y + 1,x - 1)):
                tot += 1
            if(judge(y + 1,x + 1)):
                tot += 1
            if(tot != 2):
                tmp_gujia[y, x] = [0, 0, 0]  # 移除像素 
                flag = False
    return flag
top = (-1,-1)
low = (-1,-1)
def findgujia(map):
    m = map.shape[0]
    n = map.shape[1] 
    len = 0
    for i in range(m):
        if(top != (-1,-1)):
            break
        for j in range(n):
            if(top != (-1,-1) and map[i][j] != (0,0,0)): 
                len += 1
            elif(top != (-1,-1)  and map[i][j] == (0,0,0)): 
                top = (i,len / 2)
                break
            elif(top == (-1,-1) and map[i][j] != (0,0,0)): 
                top = (i,-1)
    len = 0
    for i in range(m - 1,0,-1): 
        if(low != (-1,-1)): 
            break 
        for j in range(n): 
            if(low != (-1,-1) and map[i][j] != (0,0,0)):  
                len += 1 
            elif(low != (-1,-1)  and map[i][j] == (0,0,0)):
                low = (i,len / 2) 
                break
            elif(low == (-1,-1) and map[i][j] != (0,0,0)):
                low = (i,-1)   
    while(not bfs(top,low)) 
    gujia = tmp_gujia
    return gujia
'''
###############################裂缝提取骨架########################################
def getskeleton():
    plt.switch_backend('TkAgg')
    # 使用Otsu阈值方法进行二值化处理


    image = cv2.cvtColor(seg_image, cv2.COLOR_BGR2GRAY)
    ###cv2.imshow('image',image)
    ##cv2.waitKey(0)
    ##cv2.destroyAllWindows()
    thresh = threshold_otsu(image)
    binary = image > thresh

    binary = dilation(binary, disk(3))
    binary = median(binary, morphology.disk(5))
    binary = dilation(binary, disk(2))
    binary = median(binary,  morphology.disk(5))
    # 添加闭运算
    selem = morphology.disk(3) 
    binary = morphology.closing(binary, selem)
    skeleton = skeletonize(binary)
    skeleton_int = skeleton.astype(np.uint8) * 255
    # 显示图像
    
    plt.imshow(skeleton_int, cmap = 'gray')
    plt.axis('off')
    plt.show()
    
    return skeleton
tot = 0
def judge(a,b):
    return a>=0 and a<mm and b>=0 and b<nn and skeleton[a, b] == True
######################裂缝计算长度######################
vis = np.zeros((mm, nn), dtype=bool)
crack_length = 0

def calclen(a,b,c,d):
    #global tot
    depth1 = depth_map[a, b]
    depth2 = depth_map[c, d]
    if(math.isnan(depth1) or math.isnan(depth2) or math.isinf(depth1) or math.isinf(depth2)):
        #print("wtf?")
        return 0.621
    p1 = [(a - center_x) * depth1 / fx, (b - center_y) * depth1 / fy, depth1] 
    p2 = [(c - center_x) * depth2 / fx, (d - center_y) * depth2 / fy, depth2] 
    ret = math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]) + (p1[2] - p2[2]) * (p1[2] - p2[2]))
    #tot += ret
    #print(ret)
    ''''
    if(math.isnan(ret)):
        print(p1)
        print(p2)
        print(depth1)
    '''
    return ret
def dfs(current_point):
    #print(current_point)
    y, x = current_point
    if (not (0 <= y < mm and 0 <= x < nn)) or vis[y, x] or skeleton[y, x] == 0:
        #print(current_point)
        return 0
    vis[y, x] = True
    len = 0
    ###八方
    if (not vis[y - 1,x]) and judge(y - 1,x): 
        len += calclen(y,x,y-1,x)
        len += dfs((y - 1, x)) 
    if (not vis[y,x - 1]) and judge(y,x - 1):  
        len += calclen(y,x,y,x - 1)
        len += dfs((y, x - 1))
    if (not vis[y + 1,x]) and judge(y + 1,x): 
        #print("NO")
        len += calclen(y,x,y + 1,x)
        len += dfs((y + 1, x),)
    if (not vis[y,x + 1]) and judge(y,x + 1): 
        len += calclen(y,x,y,x + 1)
        len += dfs((y, x + 1))
    if (not vis[y + 1,x + 1]) and judge(y + 1,x + 1): 
        len += calclen(y,x,y + 1,x + 1)
        len += dfs((y + 1,x + 1))
    if (not vis[y - 1,x + 1]) and judge(y - 1,x + 1): 
        len += calclen(y,x,y - 1,x + 1)
        len += dfs((y - 1,x + 1))
    if (not vis[y - 1,x - 1]) and judge(y - 1,x - 1): 
        len += calclen(y,x,y - 1,x - 1)
        len += dfs((y - 1,x - 1))
    if (not vis[y + 1,x - 1]) and judge(y + 1,x - 1): 
        #print("YES")
        len += calclen(y,x,y + 1,x - 1)
        len += dfs((y + 1,x - 1))
    #print(len)
    return float(len)
def calc_len():
    #tt = count_true = sum(sum(1 for val in row if val) for row in skeleton)

    tp = (-1,-1)
    lw = (-1,-1)
    #print(mm,nn)
    for i in range(mm):
        if(tp != (-1,-1)):
            break
        for j in range(nn):
            #print(tp)
            #print(skeleton[i][j])
            if (skeleton[i][j] != 0): 
                tp = (i,j) 
                print(i,j)
                break
    for i in range(mm - 1,0,-1): 
        if(lw != (-1,-1)): 
            break 
        for j in range(nn): 
            if(lw != (-1,-1) and skeleton[i][j] != False):  
                lw = [i,j]
    print(tp)
    #print(lw)
    return dfs(tp)
    #print(tot / tt)

#######################计算裂缝最大宽度#######################




###深度值转化为点云计算
points_3d = []
for y in range(mm):
    for x in range(nn):
        # 计算像素点的深度值
        depth = depth_map[y, x]
        # 计算对应的3D坐标（相机坐标系下）
        point_3d = [(x - center_x) * depth / fx, (y - center_y) * depth / fy, depth]
        points_3d.append(point_3d)
 
# 将点云转换为numpy数组 
points_3d = np.array(points_3d)
skeleton = getskeleton()
print(calc_len())
# 显示结果
''''
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c=rgb_image.reshape(-1, 3)/255.0, s=1)
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
plt.show()
'''