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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from models.feed_forward_model import FeedForwardGaussianSplatting
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

class SingleViewDataset(Dataset):
    """自定义数据集，用于加载单视图图像及其多视图GT"""
    def __init__(self, dataset_path, transform=None, split='train'):
        self.dataset_path = dataset_path
        self.transform = transform
        self.split = split
        
        # 列出所有场景文件夹
        self.scenes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
        
        # 分割训练集和测试集
        if split == 'train':
            self.scenes = self.scenes[:int(len(self.scenes) * 0.8)]
        else:
            self.scenes = self.scenes[int(len(self.scenes) * 0.8):]
        
        # 收集所有输入视图和GT视图
        self.inputs = []
        self.gts = []
        
        for scene in self.scenes:
            scene_path = os.path.join(dataset_path, scene)
            views = [f for f in os.listdir(scene_path) if f.endswith('.png') or f.endswith('.jpg')]
            
            # 使用第一个视图作为输入
            input_view = views[0]
            self.inputs.append(os.path.join(scene_path, input_view))
            
            # 其余视图作为GT
            gt_views = views[1:]
            self.gts.append([os.path.join(scene_path, v) for v in gt_views])
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        # 加载输入图像
        input_path = self.inputs[idx]
        input_img = Image.open(input_path).convert('RGB')
        
        if self.transform:
            input_img = self.transform(input_img)
        
        # 随机选择一个GT视图
        gt_paths = self.gts[idx]
        gt_idx = randint(0, len(gt_paths) - 1)
        gt_path = gt_paths[gt_idx]
        gt_img = Image.open(gt_path).convert('RGB')
        
        if self.transform:
            gt_img = self.transform(gt_img)
        
        # 获取视图元数据（假设我们有相机参数文件）
        # 这里是一个简化的例子，实际上需要加载真实的相机参数
        cam_params = {
            'input_cam': self._load_camera_params(input_path),
            'gt_cam': self._load_camera_params(gt_path)
        }
        
        return {
            'input_img': input_img,
            'gt_img': gt_img,
            'cam_params': cam_params,
            'scene_name': self.scenes[idx]
        }
    
    def _load_camera_params(self, img_path):
        # 实际实现中，应该从文件中加载真实的相机参数
        # 这里返回一个占位符
        return {
            'position': torch.tensor([0.0, 0.0, 0.0]),
            'rotation': torch.tensor([0.0, 0.0, 0.0, 1.0]),
            'fov': torch.tensor([60.0])
        }

def prepare_output_and_logger(args):
    """准备输出目录和日志记录器"""
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # 设置输出文件夹
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    
    # 将配置参数写入文件
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # 创建TensorBoard的SummaryWriter
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def train_feed_forward(args, dataset_path, transform=None):
    """训练前馈高斯网络模型"""
    tb_writer = prepare_output_and_logger(args)
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 准备数据集和数据加载器
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((args.image_height, args.image_width)),
            transforms.ToTensor(),
        ])
    
    train_dataset = SingleViewDataset(dataset_path, transform=transform, split='train')
    test_dataset = SingleViewDataset(dataset_path, transform=transform, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    
    # 创建前馈高斯网络模型
    model = FeedForwardGaussianSplatting(
        input_channels=3,
        feature_dim=args.feature_dim,
        num_gaussians=args.num_gaussians,
        sh_degree=args.sh_degree
    ).to(args.device)
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 学习率调度器
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    
    # 初始化渲染管道参数
    pipe = PipelineParams(args)
    
    # 设置背景颜色
    bg_color = torch.tensor([1, 1, 1] if args.white_background else [0, 0, 0], dtype=torch.float32, device=args.device)
    
    # 训练循环
    best_psnr = 0.0
    progress_bar = tqdm(range(args.epochs), desc="Training epochs")
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            # 获取输入图像和GT图像
            input_img = batch['input_img'].to(args.device)
            gt_img = batch['gt_img'].to(args.device)
            
            # 前向传播，获取高斯参数
            gaussian_params = model(input_img)
            
            # 创建高斯模型
            gaussian_model = model.create_gaussian_model(gaussian_params)
            
            # 获取GT相机参数
            # 注意：在实际实现中，需要转换为真实的相机对象
            gt_cam = batch['cam_params']['gt_cam']
            
            # 渲染图像
            render_pkg = render(gt_cam, gaussian_model, pipe, bg_color)
            rendered_img = render_pkg["render"]
            
            # 计算损失
            l1_l = l1_loss(rendered_img, gt_img)
            ssim_l = 1.0 - ssim(rendered_img, gt_img)
            
            # 总损失
            loss = (1.0 - args.lambda_dssim) * l1_l + args.lambda_dssim * ssim_l
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            # 更新统计信息
            epoch_loss += loss.item()
            
            # 记录训练进度
            if batch_idx % args.log_interval == 0:
                if tb_writer:
                    tb_writer.add_scalar('train/loss', loss.item(), epoch * len(train_loader) + batch_idx)
                    tb_writer.add_scalar('train/l1_loss', l1_l.item(), epoch * len(train_loader) + batch_idx)
                    tb_writer.add_scalar('train/ssim_loss', ssim_l.item(), epoch * len(train_loader) + batch_idx)
        
        # 更新学习率
        lr_scheduler.step()
        
        # 验证
        if epoch % args.eval_interval == 0:
            model.eval()
            val_psnr = 0.0
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(test_loader):
                    # 获取输入图像和GT图像
                    input_img = batch['input_img'].to(args.device)
                    gt_img = batch['gt_img'].to(args.device)
                    
                    # 前向传播，获取高斯参数
                    gaussian_params = model(input_img)
                    
                    # 创建高斯模型
                    gaussian_model = model.create_gaussian_model(gaussian_params)
                    
                    # 获取GT相机参数
                    gt_cam = batch['cam_params']['gt_cam']
                    
                    # 渲染图像
                    render_pkg = render(gt_cam, gaussian_model, pipe, bg_color)
                    rendered_img = render_pkg["render"]
                    
                    # 计算损失
                    l1_l = l1_loss(rendered_img, gt_img)
                    ssim_l = 1.0 - ssim(rendered_img, gt_img)
                    loss = (1.0 - args.lambda_dssim) * l1_l + args.lambda_dssim * ssim_l
                    
                    # 计算PSNR
                    batch_psnr = psnr(rendered_img, gt_img).mean().item()
                    
                    # 更新统计信息
                    val_loss += loss.item()
                    val_psnr += batch_psnr
                    
                    # 可视化一些结果
                    if batch_idx < 5 and tb_writer:
                        tb_writer.add_images(f'val/input_{batch_idx}', input_img, epoch)
                        tb_writer.add_images(f'val/gt_{batch_idx}', gt_img, epoch)
                        tb_writer.add_images(f'val/rendered_{batch_idx}', rendered_img, epoch)
            
            # 计算平均损失和PSNR
            val_loss /= len(test_loader)
            val_psnr /= len(test_loader)
            
            # 记录验证结果
            if tb_writer:
                tb_writer.add_scalar('val/loss', val_loss, epoch)
                tb_writer.add_scalar('val/psnr', val_psnr, epoch)
            
            # 保存最佳模型
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'psnr': val_psnr,
                }, os.path.join(args.model_path, "best_model.pth"))
            
            # 打印进度
            progress_bar.set_postfix({
                'Epoch': epoch,
                'Loss': epoch_loss / len(train_loader),
                'Val Loss': val_loss,
                'Val PSNR': val_psnr,
                'Best PSNR': best_psnr
            })
        
        # 保存检查点
        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, os.path.join(args.model_path, f"checkpoint_epoch_{epoch}.pth"))
        
        progress_bar.update(1)
    
    # 保存最终模型
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(args.model_path, "final_model.pth"))
    
    progress_bar.close()
    if tb_writer:
        tb_writer.close()

if __name__ == "__main__":
    # 解析命令行参数
    parser = ArgumentParser(description="Feed-Forward Gaussian Splatting training script")
    
    # 模型参数
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--sh_degree', type=int, default=3)
    parser.add_argument('--num_gaussians', type=int, default=2000)
    parser.add_argument('--feature_dim', type=int, default=256)
    
    # 数据集参数
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--image_height', type=int, default=512)
    parser.add_argument('--image_width', type=int, default=512)
    parser.add_argument('--white_background', action='store_true', default=False)
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_step_size', type=int, default=30)
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--lambda_dssim', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    
    # 日志和保存参数
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=5)
    parser.add_argument('--save_interval', type=int, default=20)
    
    # 管道参数（与原始Gaussian Splatting兼容）
    PipelineParams.add_arguments(parser)
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available, using CPU instead")
        args.device = 'cpu'
    
    # 训练模型
    train_feed_forward(args, args.dataset_path)
    
    print("Training complete.") 