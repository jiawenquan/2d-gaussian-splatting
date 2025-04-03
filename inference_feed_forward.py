#!/usr/bin/env python3
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

import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from argparse import ArgumentParser
from models.feed_forward_model import FeedForwardGaussianSplatting
from gaussian_renderer import render
from arguments import PipelineParams
import matplotlib.pyplot as plt
from tqdm import tqdm

class CameraPose:
    """用于表示相机位姿的简单类"""
    def __init__(self, position, rotation, fov_x, fov_y, width, height, znear=0.1, zfar=100.0):
        self.position = position  # 相机位置 [x,y,z]
        self.rotation = rotation  # 相机旋转（四元数）[x,y,z,w]
        self.fov_x = fov_x  # X方向视场角（度）
        self.fov_y = fov_y  # Y方向视场角（度）
        self.width = width  # 图像宽度
        self.height = height  # 图像高度
        self.znear = znear  # 近平面
        self.zfar = zfar  # 远平面
        
        # 计算视图矩阵和投影矩阵
        self.world_view_transform = self._compute_view_matrix()
        self.full_proj_transform = self._compute_proj_matrix()
        self.camera_center = torch.tensor(position, dtype=torch.float32, device="cuda")
        
        # 添加兼容性属性
        self.image_width = width
        self.image_height = height
        self.FoVx = float(fov_x) * (3.14159265359 / 180.0)  # 转换为弧度
        self.FoVy = float(fov_y) * (3.14159265359 / 180.0)  # 转换为弧度
        
    def _compute_view_matrix(self):
        """计算相机视图矩阵"""
        # 注意：这是一个简化的实现，实际应用中需要更准确的计算
        # 从四元数计算旋转矩阵
        qx, qy, qz, qw = self.rotation
        
        # 四元数到旋转矩阵的转换
        rot_mat = torch.zeros((4, 4), dtype=torch.float32, device="cuda")
        
        # 填充3x3旋转矩阵
        rot_mat[0, 0] = 1 - 2 * (qy**2 + qz**2)
        rot_mat[0, 1] = 2 * (qx * qy - qw * qz)
        rot_mat[0, 2] = 2 * (qx * qz + qw * qy)
        
        rot_mat[1, 0] = 2 * (qx * qy + qw * qz)
        rot_mat[1, 1] = 1 - 2 * (qx**2 + qz**2)
        rot_mat[1, 2] = 2 * (qy * qz - qw * qx)
        
        rot_mat[2, 0] = 2 * (qx * qz - qw * qy)
        rot_mat[2, 1] = 2 * (qy * qz + qw * qx)
        rot_mat[2, 2] = 1 - 2 * (qx**2 + qy**2)
        
        # 设置平移部分
        rot_mat[0, 3] = -self.position[0]
        rot_mat[1, 3] = -self.position[1]
        rot_mat[2, 3] = -self.position[2]
        
        # 设置最后一行
        rot_mat[3, 3] = 1.0
        
        return rot_mat
    
    def _compute_proj_matrix(self):
        """计算投影矩阵"""
        # 注意：这是一个简化的实现，实际应用中需要更准确的计算
        aspect = self.width / self.height
        
        # 创建透视投影矩阵
        fovy_rad = self.fov_y * (3.14159265359 / 180.0)  # 转换为弧度
        f = 1.0 / np.tan(fovy_rad / 2.0)
        
        proj_mat = torch.zeros((4, 4), dtype=torch.float32, device="cuda")
        
        proj_mat[0, 0] = f / aspect
        proj_mat[1, 1] = f
        proj_mat[2, 2] = (self.zfar + self.znear) / (self.znear - self.zfar)
        proj_mat[2, 3] = (2.0 * self.zfar * self.znear) / (self.znear - self.zfar)
        proj_mat[3, 2] = -1.0
        
        return proj_mat
    
    @property
    def original_image(self):
        """为了与渲染函数兼容的空属性"""
        # 在实际应用中，这将是相机拍摄的原始图像
        return torch.zeros((3, self.height, self.width), dtype=torch.float32, device="cuda")

def create_circular_camera_path(center, radius, height, n_poses, image_width, image_height):
    """
    创建围绕中心点的圆形相机路径
    
    参数:
        center: 圆心坐标 [x,y,z]
        radius: 圆的半径
        height: 相机高度（y坐标）
        n_poses: 生成的相机姿态数量
        image_width: 图像宽度
        image_height: 图像高度
        
    返回:
        camera_poses: 相机姿态列表
    """
    camera_poses = []
    for i in range(n_poses):
        # 计算角度（0-360度）
        angle = i * (2.0 * np.pi / n_poses)
        
        # 计算位置
        x = center[0] + radius * np.cos(angle)
        z = center[2] + radius * np.sin(angle)
        position = [x, height, z]
        
        # 计算旋转（始终朝向中心）
        # 这里使用一个简化的方法计算四元数
        # 实际应用中可能需要更准确的计算
        dx, dy, dz = center[0] - x, center[1] - height, center[2] - z
        length = np.sqrt(dx*dx + dy*dy + dz*dz)
        dx, dy, dz = dx/length, dy/length, dz/length
        
        # 从朝向向量构建四元数（简化）
        # 注意：这是一个近似方法
        w = np.sqrt(1.0 + dx + dy + dz) / 2.0
        if w == 0:
            rotation = [0.0, 0.0, 0.0, 1.0]  # 默认四元数
        else:
            x_rot = (dy - dz) / (4.0 * w)
            y_rot = (dz - dx) / (4.0 * w)
            z_rot = (dx - dy) / (4.0 * w)
            rotation = [x_rot, y_rot, z_rot, w]
        
        # 创建相机姿态
        camera_pose = CameraPose(
            position=position,
            rotation=rotation,
            fov_x=60.0,
            fov_y=60.0,
            width=image_width,
            height=image_height
        )
        
        camera_poses.append(camera_pose)
    
    return camera_poses

def inference(model_path, image_path, output_dir, device="cuda", n_views=36):
    """
    使用训练好的前馈模型从单一输入图像渲染新视角
    
    参数:
        model_path: 模型检查点路径
        image_path: 输入图像路径
        output_dir: 输出目录
        device: 设备（'cuda'或'cpu'）
        n_views: 要渲染的视图数量
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    checkpoint = torch.load(model_path, map_location=device)
    
    # 确定模型参数
    # 注意：在实际应用中，这些应该从保存的配置中加载
    sh_degree = 3
    num_gaussians = 2000
    feature_dim = 256
    
    # 创建模型
    model = FeedForwardGaussianSplatting(
        input_channels=3,
        feature_dim=feature_dim,
        num_gaussians=num_gaussians,
        sh_degree=sh_degree
    ).to(device)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 初始化渲染管道
    pipe = PipelineParams(None)
    pipe.compute_cov3D_python = False
    pipe.convert_SHs_python = False
    
    # 设置背景颜色
    bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    
    # 加载和预处理输入图像
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])
    
    input_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(input_image).unsqueeze(0).to(device)
    
    # 获取图像大小
    image_width, image_height = 512, 512
    
    # 前向传播，获取高斯参数
    with torch.no_grad():
        gaussian_params = model(input_tensor)
        
        # 创建高斯模型
        gaussian_model = model.create_gaussian_model(gaussian_params)
        
        # 创建相机路径
        # 默认围绕原点的圆形路径
        camera_poses = create_circular_camera_path(
            center=[0.0, 0.0, 0.0],
            radius=4.0,
            height=0.0,
            n_poses=n_views,
            image_width=image_width,
            image_height=image_height
        )
        
        # 渲染每个视角
        print(f"Rendering {n_views} views...")
        for i, camera in enumerate(tqdm(camera_poses)):
            # 渲染图像
            render_pkg = render(camera, gaussian_model, pipe, bg_color)
            rendered_img = render_pkg["render"]
            
            # 将渲染图像转换为PIL图像并保存
            np_img = rendered_img.detach().cpu().permute(1, 2, 0).numpy()
            np_img = np.clip(np_img, 0.0, 1.0) * 255.0
            np_img = np_img.astype(np.uint8)
            pil_img = Image.fromarray(np_img)
            
            # 保存图像
            output_path = os.path.join(output_dir, f"view_{i:03d}.png")
            pil_img.save(output_path)
    
    print(f"Rendered {n_views} views and saved to {output_dir}")

def main():
    parser = ArgumentParser(description="Feed-Forward Gaussian Splatting inference script")
    
    # 必需参数
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model checkpoint")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the rendered views")
    
    # 可选参数
    parser.add_argument('--device', type=str, default='cuda', help="Device to use for inference (cuda or cpu)")
    parser.add_argument('--n_views', type=int, default=36, help="Number of views to render")
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available, using CPU instead")
        args.device = 'cpu'
    
    # 执行推理
    inference(args.model_path, args.image_path, args.output_dir, args.device, args.n_views)

if __name__ == "__main__":
    main() 