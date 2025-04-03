#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple

# 基本点云数据结构，包含点位置、颜色和法线信息
class BasicPointCloud(NamedTuple):
    points : np.array  # 点的3D坐标
    colors : np.array  # 点的颜色
    normals : np.array # 点的法线

# 几何变换函数，将点通过变换矩阵进行变换
def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    # 将点转换为齐次坐标
    points_hom = torch.cat([points, ones], dim=1)
    # 应用变换矩阵
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))
    # 透视除法，将齐次坐标转回欧几里得坐标
    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

# 构建从世界坐标到相机视图坐标的变换矩阵
def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()  # 旋转矩阵的转置
    Rt[:3, 3] = t               # 平移向量
    Rt[3, 3] = 1.0              # 齐次坐标的标准化项
    return np.float32(Rt)

# 构建从世界坐标到相机视图坐标的变换矩阵，可指定额外的平移和缩放
def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    # 计算相机到世界的变换（相机位置）
    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    # 应用额外的平移和缩放
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    # 重新计算世界到相机的变换
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

# 构建投影矩阵，用于将3D点投影到2D图像平面
def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    # 计算视锥体（view frustum）的参数
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    # 构建投影矩阵（类似于OpenGL的透视投影矩阵）
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

# 根据视场角(FOV)和像素尺寸计算焦距
def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

# 根据焦距和像素尺寸计算视场角(FOV)
def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))