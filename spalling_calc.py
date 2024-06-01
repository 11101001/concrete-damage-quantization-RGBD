import cv2
import numpy as np
from skimage.filters import threshold_otsu,median
from skimage.morphology import skeletonize,dilation,disk
from skimage import io, morphology
import math
import matplotlib.pyplot as plt
import open3d as o3d

rgb_image_path = 'E:\workspace_new\ZED_DEPTH\images\V178.png'
rgb_image = cv2.imread(rgb_image_path)

depth_image_path = 'E:\workspace_new\ZED_DEPTH\depthvalue\V178depth_values.txt'
depth_map = np.loadtxt(depth_image_path)

seg_image_path = 'E:\workspace_new\ZED_DEPTH\inference\V178(2).png'
seg_image = cv2.imread(seg_image_path) 
mm = depth_map.shape[0] 
nn = depth_map.shape[1] 
skeleton = np.zeros((mm, nn)) 


#depth_image = cv2.imread(depth_image_path,  cv2.IMREAD_GRAYSCALE)
# 定义相机内参
fx = 922.74805888  # 焦距
fy = 842.38453769 

center_x, center_y = rgb_image.shape[1] // 2, rgb_image.shape[0] // 2  # 图像中心点 
 
def calclen(a,b,c,d): 
    depth1 = depth_map[a, b] 
    depth2 = depth_map[c, d] 
    p1 = [(a - center_x) * depth1 / fx, (b - center_y) * depth1 / fy, depth1] 
    p2 = [(c - center_x) * depth2 / fx, (d - center_y) * depth2 / fy, depth2] 
    return math.sqrt((p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1]) + (p1[2] - p2[2]) * (p1[2] - p2[2]))

##################################坐标系转化

plt.switch_backend('TkAgg')#Otsu
seg_image = cv2.cvtColor(seg_image, cv2.COLOR_BGR2GRAY)
thresh = threshold_otsu(seg_image)
binary = seg_image > thresh
points1_3d = [] #### 整体
points2_3d = [] #### 损伤
points3_3d = [] #### 平面

for y in range(depth_map.shape[0]): 
    for x in range(depth_map.shape[1]): 
        depth = depth_map[y, x] 
        if depth != 'nan':
            tx = (x - center_x) * depth / fx 
            ty = (y - center_y) * depth / fy 
            point1_3d = [tx, ty, depth] 
            points1_3d.append(point1_3d)
            if binary[y, x] == 1:
                point2_3d = [tx, ty, depth]
                points2_3d.append(point2_3d)
            elif not binary[y, x]:
                point3_3d = [tx, ty, depth]
                points3_3d.append(point3_3d) 
#下采样，跑不起来的话
#downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.05)

#print(points_3d)
pcd1 = o3d.geometry.PointCloud() ## 整体 
pcd2 = o3d.geometry.PointCloud() ## 损伤
pcd3 = o3d.geometry.PointCloud() ## 平面
pcd1.points = o3d.utility.Vector3dVector(points1_3d) 
pcd2.points = o3d.utility.Vector3dVector(points2_3d) 
pcd3.points = o3d.utility.Vector3dVector(points3_3d) 
#o3d.visualization.draw_geometries([pcd2])

plane_model, inliers = pcd3.segment_plane(distance_threshold = 3, 
                                         ransac_n = 10,
                                         num_iterations = 1000)
[a, b, c, d] = plane_model# ax + by + cz + d = 0
inlier_cloud = pcd3.select_by_index(inliers)
#o3d.visualization.draw_geometries([inlier_cloud])

print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
'''
inlier_cloud = pcd1.select_by_index(inliers) 
inlier_cloud.paint_uniform_color([0, 0, 0])  
outlier_cloud = pcd1.select_by_index(inliers, invert = True) 
outlier_cloud.paint_uniform_color([0, 1, 0])  

#o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
merged_cloud = inlier_cloud + outlier_cloud

o3d.visualization.draw_geometries([inlier_cloud])

#o3d.visualization.draw_geometries([point_cloud])
#plane_model = np.array()  
''' 
# 进行投影
height = {}
def point_to_plane_distance(point, plane):
    x0, y0, z0 = point
    A, B, C, D = plane
    numerator = abs(A * x0 + B * y0 + C * z0 + D)
    denominator = math.sqrt(A**2 + B**2 + C**2)
    distance = numerator / denominator
    return distance
def round_tuple(t, precision=5):
    return tuple(round(x, precision) for x in t)

def find_closest_key(dictionary, key, tolerance=1e-4):
    for k in dictionary.keys():
        if all(abs(a - b) < tolerance for a, b in zip(k, key)):
            return k
    return None
#投影
def plane(pcd, normal_vector):
    plane_seeds = []   
    m = normal_vector[0]
    n = normal_vector[1]
    s = normal_vector[2]
    d = normal_vector[3]
    points = np.asarray(pcd.points)
    for xyz in points:
        x, y, z = xyz
        t = -(m*x + n*y + s*z + d) / (m*m + n*n + s*s)  # 计算参数方程参数
        xi = m * t + x  # 计算x的投影
        yi = n * t + y  # 计算y的投影
        zi = s * t + z  # 计算z的投影
        xi = round(xi,5)
        yi = round(yi,5)
        zi = round(zi,5)
        #print(xi,yi,zi)
        plane_seeds.append([xi, yi, zi]) 
        pt = (xi,yi,zi)
        height[round_tuple(tuple(pt))] = point_to_plane_distance(xyz,normal_vector)
    plane_cloud = o3d.geometry.PointCloud()  
    plane_cloud.points = o3d.utility.Vector3dVector(plane_seeds) 
    return plane_cloud

project_pcd = plane(pcd2, plane_model)
pcd4 = project_pcd + pcd2
#o3d.visualization.draw_geometries([pcd4])
#下采样，跑不起来的话
#downsampled_pcd = pcd.voxel_down_sample(voxel_size=0.05)
# 估计法线

pcd4.estimate_normals()

# poisson重建  
#pcd4.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius = 0.0001, max_nn = 30))
###面积
# Delaunay三角剖分
import numpy as np
import scipy.spatial

points = np.asarray(project_pcd.points)
points += np.random.normal(scale=1e-6, size=points.shape)
project_pcd.points = o3d.utility.Vector3dVector(points)
''''
print("Height dictionary keys and values:")
for key, value in height.items():
    print(f"Key: {key}, Value: {value}")
'''
with open('data.txt', 'w') as file:
    for key, value in height.items():
        file.write(f'{key}: {value}\n')

def triangle_area(p1, p2, p3):
    return 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
# 使用 Alpha 形状算法创建三角网格
'''''
alpha = 2
try:
    triangles = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(project_pcd, alpha)
    print("Triangle mesh created successfully.")
except Exception as e:
    print(f"An error occurred: {e}")
cnt = 0
total_area = 0.0
total_vol = 0.0
for i in range(len(triangles.triangles)):
    vertices = triangles.triangles[i]
    p1 = triangles.vertices[vertices[0]]
    p2 = triangles.vertices[vertices[1]]
    p3 = triangles.vertices[vertices[2]]
    # 计算三角形的面积
    cnt += 1
    print(p1,p2,p3)
    area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
    p1 = round_tuple(p1)
    p2 = round_tuple(p2)
    p3 = round_tuple(p3)
    vol = area * (height[find_closest_key(height, tuple(p1))] + height[find_closest_key(height, tuple(p2))] + height[find_closest_key(height, tuple(p3))]) / 3.0
   # print(vol)
   # print(area)
    total_vol += vol
    total_area += area
print(cnt)
print(total_area)
print(total_vol)
# 可视化
o3d.visualization.draw_geometries([project_pcd, triangles])
'''

############################################################
from scipy import spatial
import matplotlib.pyplot as plt

from scipy.spatial import Delaunay

# 基于Delaunay三角剖分的多边形面积计算
# 获取点云三维坐标
points = np.asarray(project_pcd.points)
# 获取点云XY坐标
point2d = np.c_[points[:, 0], points[:, 1]]
point_set = set(round_tuple(point) for point in points)
# Delaunay三角化
tri = spatial.Delaunay(point2d)
total_area = 0
total_vol = 0
cnt = 0
tt = 0
for simplex in tri.simplices:
    cnt+=1
    p1 = points[simplex[0]]
    p2 = points[simplex[1]]
    p3 = points[simplex[2]]
    #print(p1,p2,p3)
    area = 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))
    if(area > 1):
        print(p1,p2,p3,area)
    #print(area)
    vol = area * (height[round_tuple(p1,5)] + height[round_tuple(p2,5)] + height[ round_tuple(p3,5)]) / 3.0
    #print(vol)
    print((height[round_tuple(p1,5)] + height[round_tuple(p2,5)] + height[ round_tuple(p3,5)]) / 3.0)
    tt += (height[round_tuple(p1,5)] + height[round_tuple(p2,5)] + height[ round_tuple(p3,5)]) / 3.0
    total_area += area
    total_vol += vol
print(cnt)
print(total_area)
print(total_vol)
print(tt / cnt)
# 可视化三角化结果
plt.figure()
ax = plt.subplot(aspect="equal")
spatial.delaunay_plot_2d(tri, ax=ax)
plt.title("Point cloud delaunay  triangulation")
plt.show()

# 可视化三角化结果
#o3d.visualization.draw_geometries([project_pcd, tri])
#######################################################################

'''''
###体积
pcd4.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd4, depth=9)
mesh.remove_degenerate_triangles()
mesh.remove_duplicated_triangles()
mesh.remove_duplicated_vertices()
is_watertight = mesh.is_watertight()


# 再次检查网格是否是水密的
is_watertight = mesh.is_watertight()
mesh.compute_vertex_normals()
print(mesh)
mesh_clean = mesh.create_clean_mesh()
volume = mesh_clean.get_volume()      # 计算体积
print("体积为：", volume)
o3d.visualization.draw_geometries([mesh],
                                  zoom=0.664,
                                  front=[-0.4761, -0.4698, -0.7434],
                                  lookat=[1.8900, 3.2596, 0.9284],
                                  up=[0.2304, -0.8825, 0.4101])
# 计算体积和面积
'''