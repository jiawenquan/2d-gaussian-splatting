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

# 导入必要的库
import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

class Camera(nn.Module):
    """
    相机类，继承自PyTorch的nn.Module，表示场景中的相机。
    存储相机的内参、外参，以及与之关联的图像数据。
    用于3D-2D投影和渲染。
    """
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        """
        初始化相机对象
        
        参数:
            colmap_id: COLMAP中的相机ID
            R: 旋转矩阵，表示相机方向
            T: 平移向量，表示相机位置
            FoVx: X方向上的视场角(FOV)
            FoVy: Y方向上的视场角(FOV)
            image: 与相机关联的图像
            gt_alpha_mask: 图像的透明度蒙版（可选）
            image_name: 图像的名称
            uid: 唯一标识符
            trans: 额外的平移变换，默认为[0,0,0]
            scale: 缩放因子，默认为1.0
            data_device: 数据存储的设备，默认为"cuda"
        """
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R  # 相机旋转矩阵
        self.T = T  # 相机平移向量
        self.FoVx = FoVx  # X方向视场角
        self.FoVy = FoVy  # Y方向视场角
        self.image_name = image_name

        # 设置计算设备，优先使用指定设备，失败则回退到CUDA
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # 存储原始图像，并截取到[0,1]范围内
        self.original_image = image.clamp(0.0, 1.0) # move to device at dataloader to reduce VRAM requirement
        self.image_width = self.original_image.shape[2]  # 图像宽度
        self.image_height = self.original_image.shape[1]  # 图像高度

        # 处理透明度蒙版
        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        else:
            # self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device) # do we need this?
            self.gt_alpha_mask = None
        
        # 设置远近平面距离
        self.zfar = 100.0  # 远平面距离
        self.znear = 0.01  # 近平面距离

        self.trans = trans  # 额外平移
        self.scale = scale  # 缩放因子

        # 计算世界到相机视图的变换矩阵
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        # 计算投影矩阵
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        # 计算完整的投影变换矩阵(世界到屏幕空间)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # 计算相机中心点在世界坐标系中的位置
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    """
    轻量级相机类，不继承nn.Module，用于表示简化的相机模型。
    通常用于渲染或视图合成时的辅助相机。
    """
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        """
        初始化MiniCam对象
        
        参数:
            width: 图像宽度
            height: 图像高度
            fovy: Y方向上的视场角
            fovx: X方向上的视场角
            znear: 近平面距离
            zfar: 远平面距离
            world_view_transform: 世界到相机视图的变换矩阵
            full_proj_transform: 完整的投影变换矩阵
        """
        self.image_width = width  # 图像宽度
        self.image_height = height  # 图像高度   
        self.FoVy = fovy  # Y方向视场角
        self.FoVx = fovx  # X方向视场角
        self.znear = znear  # 近平面距离
        self.zfar = zfar  # 远平面距离
        self.world_view_transform = world_view_transform  # 世界到相机视图变换
        self.full_proj_transform = full_proj_transform  # 完整投影变换
        # 计算相机中心在世界坐标系中的位置
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

