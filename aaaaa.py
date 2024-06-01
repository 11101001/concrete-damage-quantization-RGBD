import matplotlib.pyplot as plt
import numpy as np

# 创建一个空白的图像
fig, ax = plt.subplots(figsize=(10, 5))

# 绘制原始数据
ax.text(0.1, 0.8, 'Original Data', fontsize=12, ha='center')
ax.plot([0.1, 0.1], [0.1, 0.7], 'k--')
ax.plot([0.1, 0.3], [0.1, 0.1], 'k--')
ax.add_patch(plt.Rectangle((0.05, 0.15), 0.1, 0.1, fill=None, edgecolor='red', linewidth=1))

# 绘制卷积层
ax.text(0.3, 0.8, 'Convolution', fontsize=12, ha='center')
ax.plot([0.3, 0.3], [0.1, 0.7], 'k--')
ax.plot([0.3, 0.5], [0.1, 0.1], 'k--')
ax.add_patch(plt.Rectangle((0.25, 0.15), 0.1, 0.1, fill=None, edgecolor='blue', linewidth=1))

# 绘制池化层
ax.text(0.5, 0.8, 'Pooling', fontsize=12, ha='center')
ax.plot([0.5, 0.5], [0.1, 0.7], 'k--')
ax.plot([0.5, 0.7], [0.1, 0.1], 'k--')
ax.add_patch(plt.Rectangle((0.45, 0.15), 0.1, 0.1, fill=None, edgecolor='green', linewidth=1))

# 绘制展平层
ax.text(0.7, 0.8, 'Flatten', fontsize=12, ha='center')
ax.plot([0.7, 0.7], [0.1, 0.7], 'k--')
ax.plot([0.7, 0.9], [0.1, 0.1], 'k--')
ax.add_patch(plt.Rectangle((0.65, 0.15), 0.1, 0.1, fill=None, edgecolor='purple', linewidth=1))

# 绘制全连接层
ax.text(0.9, 0.8, 'Fully connection', fontsize=12, ha='center')
ax.add_patch(plt.Rectangle((0.85, 0.15), 0.1, 0.1, fill=None, edgecolor='orange', linewidth=1))

# 绘制输出节点
ax.text(1.1, 0.8, 'Output nodes', fontsize=12, ha='center')
ax.add_patch(plt.Circle((1.05, 0.2), 0.05, fill=None, edgecolor='black', linewidth=1))

# 设置轴的范围和隐藏轴
ax.set_xlim(0, 1.2)
ax.set_ylim(0, 1)
ax.axis('off')

# 显示图像
plt.show()