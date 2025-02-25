# 导入必要的库
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 定义相机内参和外参，这里只是示例，你需要根据你的实际情况进行修改
# 相机内参矩阵，包含焦距和主点坐标
K = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]])
# 相机外参矩阵，包含旋转和平移
R = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) # 单位矩阵表示无旋转
t = np.array([0, 0, -1000]) # 表示相机在z轴负方向上距离原点1000单位
# 将旋转矩阵和平移向量合并为一个变换矩阵
T = np.hstack((R, t.reshape(3, 1)))

# 加载3D关节数据，假设它的形状是(N, J, 3)，其中N是帧数，J是关节数，3是坐标维度
output_3d = np.load('output.npy')
# 创建一个空的数组来存储投影后的2D关节数据，它的形状是(N, J, 2)
output_2d = np.zeros((output_3d.shape[0], output_3d.shape[1], 2))

# 对每一帧的3D关节数据进行投影
for i in range(output_3d.shape[0]):
    # 获取当前帧的3D关节数据，它的形状是(J, 3)
    joints_3d = output_3d[i]
    # 将其转换为齐次坐标，即在最后一维添加一个1，它的形状是(J, 4)
    joints_3d_homo = np.hstack((joints_3d, np.ones((joints_3d.shape[0], 1))))
    # 使用相机内参和外参对其进行变换，得到相机坐标系下的齐次坐标，它的形状是(J, 4)
    joints_cam_homo = (K @ T @ joints_3d_homo.T).T
    # 使用透视除法将其转换为归一化坐标，即除以最后一维，它的形状是(J, 4)
    joints_norm_homo = joints_cam_homo / joints_cam_homo[:, -1:]
    # 取前两维作为像素坐标，它的形状是(J, 2)
    joints_2d = joints_norm_homo[:, :2]
    # 将其存储到输出数组中
    output_2d[i] = joints_2d

# 将输出数组保存为npy文件
np.save('output_2d.npy', output_2d)