import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import glob
from skimage.morphology import binary_dilation, disk
import argparse

import trimesh
from pathlib import Path
import subprocess

import sys
import render_utils as rend_util
from tqdm import tqdm

def cull_scan(scan, mesh_path, result_mesh_file, instance_dir):
    """
    裁剪网格模型，使其与图像掩码保持一致
    
    参数:
        scan: 扫描ID
        mesh_path: 输入网格模型的路径
        result_mesh_file: 输出结果网格的路径
        instance_dir: 包含相机参数和掩码的实例目录
    """
    
    # 加载相机位姿信息
    image_dir = '{0}/images'.format(instance_dir)
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    n_images = len(image_paths)  # 图像数量
    cam_file = '{0}/cameras.npz'.format(instance_dir)
    camera_dict = np.load(cam_file)  # 加载相机参数文件
    # 提取缩放矩阵和世界矩阵
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]

    # 存储所有相机的内参和位姿
    intrinsics_all = []
    pose_all = []
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        P = world_mat @ scale_mat  # 计算投影矩阵
        P = P[:3, :4]  # 获取3x4部分
        # 从投影矩阵分解得到内参和位姿
        intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
        intrinsics_all.append(torch.from_numpy(intrinsics).float())
        pose_all.append(torch.from_numpy(pose).float())
    
    # 加载掩码图像
    mask_dir = '{0}/mask'.format(instance_dir)
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    masks = []
    for p in mask_paths:
        mask = cv2.imread(p)
        masks.append(mask)

    # 硬编码的图像尺寸
    W, H = 1600, 1200

    # 加载网格模型
    mesh = trimesh.load(mesh_path)
    
    # 获取网格顶点
    vertices = mesh.vertices

    # 投影和过滤
    # 将顶点转换为GPU张量
    vertices = torch.from_numpy(vertices).cuda()
    # 添加齐次坐标（w=1）
    vertices = torch.cat((vertices, torch.ones_like(vertices[:, :1])), dim=-1)
    # 转置为形状 [4, N]
    vertices = vertices.permute(1, 0)
    vertices = vertices.float()

    # 对每个图像采样掩码
    sampled_masks = []
    for i in tqdm(range(n_images),  desc="Culling mesh given masks"):
        pose = pose_all[i]
        w2c = torch.inverse(pose).cuda()  # 世界坐标到相机坐标的变换矩阵
        intrinsic = intrinsics_all[i].cuda()  # 相机内参

        with torch.no_grad():
            # 变换并投影顶点到相机空间
            cam_points = intrinsic @ w2c @ vertices
            # 将3D点投影到2D像素坐标
            pix_coords = cam_points[:2, :] / (cam_points[2, :].unsqueeze(0) + 1e-6)
            pix_coords = pix_coords.permute(1, 0)
            # 归一化像素坐标到 [0,1] 范围
            pix_coords[..., 0] /= W - 1
            pix_coords[..., 1] /= H - 1
            # 归一化到 [-1,1] 范围，用于grid_sample
            pix_coords = (pix_coords - 0.5) * 2
            # 检查坐标是否在有效范围内
            valid = ((pix_coords > -1. ) & (pix_coords < 1.)).all(dim=-1).float()
            
            # 扩展掩码，类似于unisurf的方法
            maski = masks[i][:, :, 0].astype(np.float32) / 256.  # 归一化掩码值
            # 使用形态学操作扩展掩码
            maski = torch.from_numpy(binary_dilation(maski, disk(24))).float()[None, None].cuda()

            # 使用grid_sample在掩码上采样
            sampled_mask = F.grid_sample(maski, pix_coords[None, None], mode='nearest', padding_mode='zeros', align_corners=True)[0, -1, 0]

            # 将无效点也标记为掩码外
            sampled_mask = sampled_mask + (1. - valid)
            sampled_masks.append(sampled_mask)

    # 堆叠所有采样的掩码
    sampled_masks = torch.stack(sampled_masks, -1)
    # 过滤：只保留在所有视图中都在掩码内的点
    
    mask = (sampled_masks > 0.).all(dim=-1).cpu().numpy()
    # 如果一个面的所有顶点都在掩码内，则保留该面
    face_mask = mask[mesh.faces].all(axis=1)

    # 更新网格的顶点和面
    mesh.update_vertices(mask)
    mesh.update_faces(face_mask)
    
    # 将顶点变换到世界坐标系 
    scale_mat = scale_mats[0]
    mesh.vertices = mesh.vertices * scale_mat[0, 0] + scale_mat[:3, 3][None]
    # 导出结果网格
    mesh.export(result_mesh_file)
    del mesh
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Arguments to evaluate the mesh.'
    )

    # 定义命令行参数
    parser.add_argument('--input_mesh', type=str,  help='path to the mesh to be evaluated')  # 待评估的网格模型路径
    parser.add_argument('--scan_id', type=str,  help='scan id of the input mesh')  # 输入网格的扫描ID
    parser.add_argument('--output_dir', type=str, default='evaluation_results_single', help='path to the output folder')  # 输出文件夹路径
    parser.add_argument('--mask_dir', type=str,  default='mask', help='path to uncropped mask')  # 未裁剪掩码的路径
    parser.add_argument('--DTU', type=str,  default='Offical_DTU_Dataset', help='path to the GT DTU point clouds')  # 官方DTU点云数据集路径
    args = parser.parse_args()

    Offical_DTU_Dataset = args.DTU
    out_dir = args.output_dir
    # 创建输出目录
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    scan = args.scan_id
    ply_file = args.input_mesh
    print("cull mesh ....")
    # 定义裁剪后的网格文件路径
    result_mesh_file = os.path.join(out_dir, "culled_mesh.ply")
    # 执行网格裁剪
    cull_scan(scan, ply_file, result_mesh_file, instance_dir=os.path.join(args.mask_dir, f'scan{args.scan_id}'))

    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建评估命令并执行
    cmd = f"python {script_dir}/eval.py --data {result_mesh_file} --scan {scan} --mode mesh --dataset_dir {Offical_DTU_Dataset} --vis_out_dir {out_dir}"
    os.system(cmd)