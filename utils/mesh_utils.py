#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#

import torch
import numpy as np
import os
import math
from tqdm import tqdm
from utils.render_utils import save_img_f32, save_img_u8
from functools import partial
import open3d as o3d
import trimesh

def post_process_mesh(mesh, cluster_to_keep=1000):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    import copy
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)  # 创建网格的深拷贝，避免修改原始网格
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
            # 聚类连接的三角形，识别连接组件
            triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)  # 将三角形聚类转换为NumPy数组
    cluster_n_triangles = np.asarray(cluster_n_triangles)  # 每个聚类中的三角形数量
    cluster_area = np.asarray(cluster_area)  # 每个聚类的面积
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]  # 获取要保留的最小聚类大小
    n_cluster = max(n_cluster, 50) # 过滤小于50个三角形的网格，确保最小网格规模
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster  # 标记要移除的三角形
    mesh_0.remove_triangles_by_mask(triangles_to_remove)  # 根据掩码移除三角形
    mesh_0.remove_unreferenced_vertices()  # 移除未被引用的顶点
    mesh_0.remove_degenerate_triangles()  # 移除退化的三角形（面积为零的三角形）
    print("num vertices raw {}".format(len(mesh.vertices)))  # 打印原始顶点数量
    print("num vertices post {}".format(len(mesh_0.vertices)))  # 打印处理后的顶点数量
    return mesh_0

def to_cam_open3d(viewpoint_stack):
    """
    将视点栈转换为Open3D相机参数列表
    
    参数:
    viewpoint_stack - 包含相机视点信息的列表
    
    返回:
    camera_traj - Open3D相机参数列表
    """
    camera_traj = []
    for i, viewpoint_cam in enumerate(viewpoint_stack):
        W = viewpoint_cam.image_width  # 获取图像宽度
        H = viewpoint_cam.image_height  # 获取图像高度
        # 创建从NDC坐标到像素坐标的变换矩阵
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, 0, 1]]).float().cuda().T
        # 计算内部参数矩阵
        intrins =  (viewpoint_cam.projection_matrix @ ndc2pix)[:3,:3].T
        # 创建Open3D的针孔相机内参
        intrinsic=o3d.camera.PinholeCameraIntrinsic(
            width=viewpoint_cam.image_width,
            height=viewpoint_cam.image_height,
            cx = intrins[0,2].item(),  # 主点x坐标
            cy = intrins[1,2].item(),  # 主点y坐标
            fx = intrins[0,0].item(),  # x方向焦距
            fy = intrins[1,1].item()   # y方向焦距
        )

        # 获取外部参数矩阵（世界到相机的变换）
        extrinsic=np.asarray((viewpoint_cam.world_view_transform.T).cpu().numpy())
        camera = o3d.camera.PinholeCameraParameters()
        camera.extrinsic = extrinsic  # 设置外参
        camera.intrinsic = intrinsic  # 设置内参
        camera_traj.append(camera)

    return camera_traj


class GaussianExtractor(object):
    def __init__(self, gaussians, render, pipe, bg_color=None):
        """
        一个用于提取2DGS场景属性的类
        
        参数:
        gaussians - 高斯模型
        render - 渲染函数
        pipe - 渲染管道
        bg_color - 背景颜色，默认为黑色 [0, 0, 0]
        
        使用示例:
        >>> gaussExtrator = GaussianExtractor(gaussians, render, pipe)
        >>> gaussExtrator.reconstruction(view_points)
        >>> mesh = gaussExtractor.export_mesh_bounded(...)
        """
        if bg_color is None:
            bg_color = [0, 0, 0]  # 如果未指定背景颜色，则默认为黑色
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")  # 将背景颜色转换为CUDA张量
        self.gaussians = gaussians  # 存储高斯模型
        self.render = partial(render, pipe=pipe, bg_color=background)  # 创建带有固定参数的渲染函数
        self.clean()  # 初始化时清理状态

    @torch.no_grad()
    def clean(self):
        """
        清理实例状态，重置所有列表
        """
        self.depthmaps = []  # 存储深度图
        # self.alphamaps = []  # 存储透明度图（已注释）
        self.rgbmaps = []  # 存储RGB图
        # self.normals = []  # 存储法线图（已注释）
        # self.depth_normals = []  # 存储深度法线图（已注释）
        self.viewpoint_stack = []  # 存储视点栈

    @torch.no_grad()
    def reconstruction(self, viewpoint_stack):
        """
        根据给定的相机视点重建辐射场
        
        参数:
        viewpoint_stack - 包含相机视点的列表
        """
        self.clean()  # 清理之前的状态
        self.viewpoint_stack = viewpoint_stack  # 保存视点栈
        for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="reconstruct radiance fields"):
            render_pkg = self.render(viewpoint_cam, self.gaussians)  # 渲染当前视点
            rgb = render_pkg['render']  # 获取渲染的RGB图
            alpha = render_pkg['rend_alpha']  # 获取渲染的透明度图
            normal = torch.nn.functional.normalize(render_pkg['rend_normal'], dim=0)  # 获取并归一化法线图
            depth = render_pkg['surf_depth']  # 获取深度图
            depth_normal = render_pkg['surf_normal']  # 获取深度法线图
            self.rgbmaps.append(rgb.cpu())  # 保存RGB图到CPU内存
            self.depthmaps.append(depth.cpu())  # 保存深度图到CPU内存
            # self.alphamaps.append(alpha.cpu())  # 保存透明度图（已注释）
            # self.normals.append(normal.cpu())  # 保存法线图（已注释）
            # self.depth_normals.append(depth_normal.cpu())  # 保存深度法线图（已注释）
        
        # self.rgbmaps = torch.stack(self.rgbmaps, dim=0)  # 将RGB图堆叠为一个张量（已注释）
        # self.depthmaps = torch.stack(self.depthmaps, dim=0)  # 将深度图堆叠为一个张量（已注释）
        # self.alphamaps = torch.stack(self.alphamaps, dim=0)  # 将透明度图堆叠为一个张量（已注释）
        # self.depth_normals = torch.stack(self.depth_normals, dim=0)  # 将深度法线图堆叠为一个张量（已注释）
        self.estimate_bounding_sphere()  # 估计边界球

    def estimate_bounding_sphere(self):
        """
        根据相机位姿估计场景的边界球
        """
        from utils.render_utils import transform_poses_pca, focus_point_fn
        torch.cuda.empty_cache()  # 清理GPU内存
        # 将相机的世界到视图变换转换为相机到世界变换（逆矩阵）
        c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in self.viewpoint_stack])
        poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])  # 应用坐标系变换
        center = (focus_point_fn(poses))  # 计算焦点（场景中心）
        # 计算半径为相机到中心的最小距离
        self.radius = np.linalg.norm(c2ws[:,:3,3] - center, axis=-1).min()
        self.center = torch.from_numpy(center).float().cuda()  # 将中心转换为CUDA张量
        print(f"The estimated bounding radius is {self.radius:.2f}")  # 打印估计的边界半径
        print(f"Use at least {2.0 * self.radius:.2f} for depth_trunc")  # 建议设置深度截断值

    @torch.no_grad()
    def extract_mesh_bounded(self, voxel_size=0.004, sdf_trunc=0.02, depth_trunc=3, mask_backgrond=True):
        """
        执行TSDF融合，提取有界场景的网格，论文中使用的方法
        
        参数:
        voxel_size - 体素大小
        sdf_trunc - 符号距离场截断值
        depth_trunc - 最大深度范围，应根据场景尺度调整
        mask_backgrond - 是否遮蔽背景，仅在数据集有掩码时有效
        
        返回:
        o3d.mesh - Open3D网格对象
        """
        print("Running tsdf volume integration ...")
        print(f'voxel_size: {voxel_size}')
        print(f'sdf_trunc: {sdf_trunc}')
        print(f'depth_truc: {depth_trunc}')

        # 创建可扩展的TSDF体素体积
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length= voxel_size,  # 体素大小
            sdf_trunc=sdf_trunc,  # SDF截断值
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8  # 颜色类型为RGB8
        )

        # 遍历所有相机视点进行TSDF集成
        for i, cam_o3d in tqdm(enumerate(to_cam_open3d(self.viewpoint_stack)), desc="TSDF integration progress"):
            rgb = self.rgbmaps[i]  # 获取RGB图
            depth = self.depthmaps[i]  # 获取深度图
            
            # 如果提供了掩码，使用它
            if mask_backgrond and (self.viewpoint_stack[i].gt_alpha_mask is not None):
                depth[(self.viewpoint_stack[i].gt_alpha_mask < 0.5)] = 0  # 使用掩码将背景深度置零

            # 创建Open3D的RGBD图像
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.asarray(np.clip(rgb.permute(1,2,0).cpu().numpy(), 0.0, 1.0) * 255, order="C", dtype=np.uint8)),  # 颜色图像
                o3d.geometry.Image(np.asarray(depth.permute(1,2,0).cpu().numpy(), order="C")),  # 深度图像
                depth_trunc = depth_trunc,  # 深度截断值
                convert_rgb_to_intensity=False,  # 不转换RGB为灰度
                depth_scale = 1.0  # 深度缩放因子
            )

            # 将RGBD图像集成到TSDF体积中
            volume.integrate(rgbd, intrinsic=cam_o3d.intrinsic, extrinsic=cam_o3d.extrinsic)

        # 从TSDF体积中提取三角网格
        mesh = volume.extract_triangle_mesh()
        return mesh

    @torch.no_grad()
    def extract_mesh_unbounded(self, resolution=1024):
        """
        实验性功能，从无界场景中提取网格，尚未在所有数据集上完全测试
        
        参数:
        resolution - 体素网格分辨率
        
        返回:
        o3d.mesh - Open3D网格对象
        """
        def contract(x):
            """收缩函数，将无界空间映射到有界空间"""
            mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]  # 计算向量范数
            return torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag))  # 当范数小于1时保持不变，否则收缩
        
        def uncontract(y):
            """解收缩函数，将有界空间映射回无界空间"""
            mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]  # 计算向量范数
            return torch.where(mag < 1, y, (1 / (2-mag) * (y/mag)))  # 当范数小于1时保持不变，否则解收缩
        
        def compute_sdf_perframe(i, points, depthmap, rgbmap, viewpoint_cam):
            """
            计算每帧的SDF值
            
            参数:
            i - 帧索引
            points - 查询点
            depthmap - 深度图
            rgbmap - RGB图
            viewpoint_cam - 视点相机
            
            返回:
            sdf - 符号距离场值
            sampled_rgb - 采样的RGB值
            mask_proj - 投影掩码
            """
            # 将点投影到相机空间
            new_points = torch.cat([points, torch.ones_like(points[...,:1])], dim=-1) @ viewpoint_cam.full_proj_transform
            z = new_points[..., -1:]  # 深度值
            pix_coords = (new_points[..., :2] / new_points[..., -1:])  # 归一化坐标
            # 检查点是否在图像平面内且位于相机前方
            mask_proj = ((pix_coords > -1. ) & (pix_coords < 1.) & (z > 0)).all(dim=-1)
            # 使用双线性插值采样深度值
            sampled_depth = torch.nn.functional.grid_sample(depthmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(-1, 1)
            # 使用双线性插值采样RGB值
            sampled_rgb = torch.nn.functional.grid_sample(rgbmap.cuda()[None], pix_coords[None, None], mode='bilinear', padding_mode='border', align_corners=True).reshape(3,-1).T
            sdf = (sampled_depth-z)  # 计算SDF值：采样深度 - 点深度
            return sdf, sampled_rgb, mask_proj

        def compute_unbounded_tsdf(samples, inv_contraction, voxel_size, return_rgb=False):
            """
            融合所有帧，在收缩空间上执行自适应SDF截断
            
            参数:
            samples - 采样点
            inv_contraction - 解收缩函数
            voxel_size - 体素大小
            return_rgb - 是否返回RGB值
            
            返回:
            tsdfs - TSDF值
            rgbs - RGB值（如果return_rgb=True）
            """
            if inv_contraction is not None:
                # 检查点是否在单位球外
                mask = torch.linalg.norm(samples, dim=-1) > 1
                # 自适应SDF截断
                sdf_trunc = 5 * voxel_size * torch.ones_like(samples[:, 0])
                sdf_trunc[mask] *= 1/(2-torch.linalg.norm(samples, dim=-1)[mask].clamp(max=1.9))
                samples = inv_contraction(samples)  # 应用解收缩变换
            else:
                sdf_trunc = 5 * voxel_size  # 固定SDF截断值

            tsdfs = torch.ones_like(samples[:,0]) * 1  # 初始化TSDF值
            rgbs = torch.zeros((samples.shape[0], 3)).cuda()  # 初始化RGB值

            weights = torch.ones_like(samples[:,0])  # 初始化权重
            # 遍历所有视点
            for i, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="TSDF integration progress"):
                # 计算当前视点的SDF值
                sdf, rgb, mask_proj = compute_sdf_perframe(i, samples,
                    depthmap = self.depthmaps[i],
                    rgbmap = self.rgbmaps[i],
                    viewpoint_cam=self.viewpoint_stack[i],
                )

                # 体积集成
                sdf = sdf.flatten()
                # 仅考虑投影内且SDF值大于负截断值的点
                mask_proj = mask_proj & (sdf > -sdf_trunc)
                # 将SDF值截断到[-1, 1]范围
                sdf = torch.clamp(sdf / sdf_trunc, min=-1.0, max=1.0)[mask_proj]
                w = weights[mask_proj]  # 获取当前权重
                wp = w + 1  # 更新权重
                # 加权平均更新TSDF值
                tsdfs[mask_proj] = (tsdfs[mask_proj] * w + sdf) / wp
                # 加权平均更新RGB值
                rgbs[mask_proj] = (rgbs[mask_proj] * w[:,None] + rgb[mask_proj]) / wp[:,None]
                # 更新权重
                weights[mask_proj] = wp
            
            if return_rgb:
                return tsdfs, rgbs

            return tsdfs

        # 定义归一化和反归一化函数
        normalize = lambda x: (x - self.center) / self.radius  # 将点归一化到单位球
        unnormalize = lambda x: (x * self.radius) + self.center  # 将点从单位球转回原始空间
        inv_contraction = lambda x: unnormalize(uncontract(x))  # 组合解收缩和反归一化

        N = resolution  # 网格分辨率
        voxel_size = (self.radius * 2 / N)  # 计算体素大小
        print(f"Computing sdf gird resolution {N} x {N} x {N}")
        print(f"Define the voxel_size as {voxel_size}")
        # 定义SDF函数
        sdf_function = lambda x: compute_unbounded_tsdf(x, inv_contraction, voxel_size)
        # 导入行进立方体工具
        from utils.mcube_utils import marching_cubes_with_contraction
        # 计算高斯点的收缩半径
        R = contract(normalize(self.gaussians.get_xyz)).norm(dim=-1).cpu().numpy()
        R = np.quantile(R, q=0.95)  # 取95%分位数作为边界
        R = min(R+0.01, 1.9)  # 添加一点余量，但不超过1.9（收缩空间边界）

        # 在收缩空间中执行行进立方体算法提取等值面
        mesh = marching_cubes_with_contraction(
            sdf=sdf_function,  # SDF函数
            bounding_box_min=(-R, -R, -R),  # 边界框最小点
            bounding_box_max=(R, R, R),  # 边界框最大点
            level=0,  # 等值面级别
            resolution=N,  # 分辨率
            inv_contraction=inv_contraction,  # 解收缩函数
        )
        
        # 为网格上色
        torch.cuda.empty_cache()  # 清理GPU内存
        mesh = mesh.as_open3d  # 转换为Open3D网格
        print("texturing mesh ... ")
        # 计算网格顶点的RGB值
        _, rgbs = compute_unbounded_tsdf(torch.tensor(np.asarray(mesh.vertices)).float().cuda(), inv_contraction=None, voxel_size=voxel_size, return_rgb=True)
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgbs.cpu().numpy())  # 设置顶点颜色
        return mesh

    @torch.no_grad()
    def export_image(self, path):
        """
        导出渲染的图像到指定路径
        
        参数:
        path - 输出路径
        """
        render_path = os.path.join(path, "renders")  # 渲染图像路径
        gts_path = os.path.join(path, "gt")  # 真实图像路径
        vis_path = os.path.join(path, "vis")  # 可视化路径
        os.makedirs(render_path, exist_ok=True)  # 创建渲染图像目录
        os.makedirs(vis_path, exist_ok=True)  # 创建可视化目录
        os.makedirs(gts_path, exist_ok=True)  # 创建真实图像目录
        
        # 遍历所有视点
        for idx, viewpoint_cam in tqdm(enumerate(self.viewpoint_stack), desc="export images"):
            gt = viewpoint_cam.original_image[0:3, :, :]  # 获取原始图像的RGB通道
            # 保存真实图像
            save_img_u8(gt.permute(1,2,0).cpu().numpy(), os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            # 保存渲染的RGB图像
            save_img_u8(self.rgbmaps[idx].permute(1,2,0).cpu().numpy(), os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            # 保存深度图
            save_img_f32(self.depthmaps[idx][0].cpu().numpy(), os.path.join(vis_path, 'depth_{0:05d}'.format(idx) + ".tiff"))
            # save_img_u8(self.normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'normal_{0:05d}'.format(idx) + ".png"))  # 保存法线图（已注释）
            # save_img_u8(self.depth_normals[idx].permute(1,2,0).cpu().numpy() * 0.5 + 0.5, os.path.join(vis_path, 'depth_normal_{0:05d}'.format(idx) + ".png"))  # 保存深度法线图（已注释）
