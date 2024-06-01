import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------读取深度图像--------------------------------
depth = o3d.t.io.read_image('E:\workspace_new\ZED_DEPTH\dimages\V008.png')
# ------------------------------设置相机内参--------------------------------
intrinsic = o3d.core.Tensor([[922.74805888,0,861.07449695],[0,842.38453769,738.75205406],[0,0,1]]) 
# ------------------------------可视化深度图--------------------------------
fig, axs = plt.subplots()
axs.imshow(np.asarray(depth.to_legacy()))
plt.show()
# ------------------------------深度图转点云-------------------------------- 
pcd = o3d.t.geometry.PointCloud.create_from_depth_image(depth,              
                                                        intrinsic,          
                                                        depth_scale=5000.0, 
                                                        depth_max=10.0)     
# --------------------------------保存结果---------------------------------- 
o3d.io.write_point_cloud("depth2cloud.pcd", pcd.to_legacy())
# -------------------------------结果可视化---------------------------------
o3d.visualization.draw([pcd])

