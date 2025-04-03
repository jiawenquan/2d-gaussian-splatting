import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, cv2
import matplotlib.pyplot as plt
import math

# 将深度图转换为3D点云
def depths_to_points(view, depthmap):
    # 获取相机到世界的变换矩阵（逆变换）
    c2w = (view.world_view_transform.T).inverse()
    # 获取图像宽度和高度
    W, H = view.image_width, view.image_height
    # 创建从NDC坐标到像素坐标的变换矩阵
    ndc2pix = torch.tensor([
        [W / 2, 0, 0, (W) / 2],
        [0, H / 2, 0, (H) / 2],
        [0, 0, 0, 1]]).float().cuda().T
    # 计算投影矩阵，用于将像素坐标投影到相机坐标
    projection_matrix = c2w.T @ view.full_proj_transform
    # 计算内参矩阵
    intrins = (projection_matrix @ ndc2pix)[:3,:3].T
    
    # 生成图像平面上的网格坐标
    grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
    # 构建齐次坐标点 [x, y, 1]
    points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
    # 计算从相机出发的光线方向
    rays_d = points @ intrins.inverse().T @ c2w[:3,:3].T
    # 获取相机中心位置
    rays_o = c2w[:3,3]
    # 根据深度值计算3D点的位置：相机位置 + 深度 * 光线方向
    points = depthmap.reshape(-1, 1) * rays_d + rays_o
    return points

# 从深度图计算法线图
def depth_to_normal(view, depth):
    """
        view: view camera
        depth: depthmap 
    """
    # 将深度图转换为三维点云
    points = depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
    # 初始化法线输出
    output = torch.zeros_like(points)
    # 计算x方向的点差异（相邻两点之间的差）
    dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
    # 计算y方向的点差异
    dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
    # 通过两个方向的差异向量的叉积计算法线，并归一化
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
    # 将计算好的法线填充到输出张量（不包括边缘像素）
    output[1:-1, 1:-1, :] = normal_map
    return output