import open3d as o3d
import numpy as np
from PIL import Image

data = np.loadtxt("E:\workspace_new\ZED_DEPTH\pointcloud\V538.txt")
points = data[:, :3] 
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
vis = o3d.visualization.Visualizer()
vis.create_window()

#添加点云到可视化窗口  
vis.add_geometry(pcd) 

# 设置背景颜色为灰色
vis.get_render_option().background_color = np.asarray([0.5, 0.5, 0.5])

# 设置点的大小为1
vis.get_render_option().point_size = 1

# 显示点云
vis.run()
vis.destroy_window()



pcd.paint_uniform_color(np.array([0, 1, 0]))  # 绿色
o3d.visualization.draw_geometries([pcd])


plane_model, inliers = pcd.segment_plane(distance_threshold = 0.05,
                                         ransac_n = 10,
                                         num_iterations = 1000)
[a, b, c, d] = plane_model 
print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

inlier_cloud = pcd.select_by_index(inliers)
inlier_cloud.paint_uniform_color([0, 0, 0])
outlier_cloud = pcd.select_by_index(inliers, invert=True)
outlier_cloud.paint_uniform_color([0, 1, 0])
#o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
merged_cloud = inlier_cloud + outlier_cloud
o3d.visualization.draw_geometries([inlier_cloud])


### projection part
def point_cloud_plane_project(cloud, coefficients):
    """
    点云投影到平面
    :param cloud:输入点云
    :param coefficients: 待投影的平面
    :return: 投影后的点云
    """
    # 获取平面系数
    A = coefficients[0]
    B = coefficients[1]
    C = coefficients[2]
    D = coefficients[3]
    # 构建投影函数
    Xcoff = np.array([B * B + C * C, -A * B, -A * C])
    Ycoff = np.array([-B * A, A * A + C * C, -B * C])
    Zcoff = np.array([-A * C, -B * C, A * A + B * B])
    # 三维坐标执行投影
    points = np.asarray(cloud.points)
    xp = np.dot(points, Xcoff) - A * D
    yp = np.dot(points, Ycoff) - B * D
    zp = np.dot(points, Zcoff) - C * D
    project_points = np.c_[xp, yp, zp]  # 投影后的三维坐标
    project_cloud = o3d.geometry.PointCloud()  # 使用numpy生成点云
    project_cloud.points = o3d.utility.Vector3dVector(project_points)
    project_cloud.colors = pcd.colors  # 获取投影前对应的颜色赋值给投影后的点
    return project_cloud

projected_cloud = point_cloud_plane_project(inlier_cloud, plane_model)
#o3d.io.write_point_cloud("project_cloud.pcd", projected_cloud)
o3d.visualization.draw_geometries([projected_cloud], window_name="点云投影到平面", 
                                  width = 900, height = 900,
                                  left = 50, top = 50,
                                  mesh_show_back_face = False)

'''
import open3d as o3d


print("Testing mesh in Open3D...")
mesh = o3d.io.read_triangle_mesh("data//UV.ply")
print(mesh)  # 打印点数和三角面数
print("Computing normal and rendering it.")
mesh.compute_vertex_normals()
mesh.paint_uniform_color([1, 0.7, 0])
area = mesh.get_surface_area()  # 计算表面积
# volume = mesh.get_volume()      # 计算体积
print("表面积为：", area)
# print("体积为：", volume)
o3d.visualization.draw_geometries([mesh], window_name="计算点云模型的表面积并显示",
                                  width=1024, height=768,
                                  left=50, top=50,
                                  mesh_show_back_face=False)  # 显示点云模型




#########
vis = o3d.visualization.Visualizer()
vis.create_window(width = 640, height = 480)

# 添加点云到渲染器
vis.add_geometry(inlier_cloud)

# 渲染并获取图像
vis.poll_events()
vis.update_renderer()
image = vis.capture_screen_float_buffer(True)

# 关闭窗口
vis.destroy_window()

# 将浮点数图像转换为8位图像
image = (np.asarray(image) * 255).astype(np.uint8)
image = Image.fromarray(image)

# 保存为JPEG图片
image.save("point_cloud_image.jpg")

# 显示图像
image.show()
#####
'''