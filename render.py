# 版权声明：该软件属于 Inria GRAPHDECO 研究组，仅供非商业、研究和评估使用
# 版权所有 (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from scene import Scene  # 导入场景类，用于管理3D场景数据
import os
from tqdm import tqdm  # 导入进度条库
from os import makedirs
from gaussian_renderer import render  # 导入高斯渲染函数
import torchvision
from utils.general_utils import safe_state  # 导入安全保存状态的工具函数
from argparse import ArgumentParser  # 命令行参数解析库
from arguments import ModelParams, PipelineParams, get_combined_args  # 导入模型参数、渲染管道参数类
from gaussian_renderer import GaussianModel  # 导入高斯模型类
from utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh  # 导入网格处理工具
from utils.render_utils import generate_path, create_videos  # 导入路径生成和视频创建工具

import open3d as o3d  # 导入Open3D库，用于3D数据处理

if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)  # 添加模型相关参数
    pipeline = PipelineParams(parser)  # 添加渲染管道相关参数
    parser.add_argument("--iteration", default=-1, type=int)  # 指定加载模型的迭代次数，默认为最后一次
    parser.add_argument("--skip_train", action="store_true")  # 是否跳过训练集渲染
    parser.add_argument("--skip_test", action="store_true")  # 是否跳过测试集渲染
    parser.add_argument("--skip_mesh", action="store_true")  # 是否跳过网格导出
    parser.add_argument("--quiet", action="store_true")  # 是否安静模式
    parser.add_argument("--render_path", action="store_true")  # 是否渲染相机路径
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')  # 用于TSDF的体素大小
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')  # TSDF的最大深度范围
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')  # TSDF的截断值
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')  # 导出的连通集群数量
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')  # 是否使用无边界模式生成网格
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')  # 无边界网格提取的分辨率
    args = get_combined_args(parser)  # 获取解析后的参数
    print("Rendering " + args.model_path)  # 打印要渲染的模型路径


    # 加载数据集、迭代次数和渲染管道参数
    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)  # 初始化高斯模型，使用数据集中的球谐函数度数
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)  # 创建场景，加载指定迭代的模型
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]  # 根据数据集设置背景颜色
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")  # 将背景颜色转换为CUDA张量
    
    # 设置训练和测试输出目录
    train_dir = os.path.join(args.model_path, 'train', "ours_{}".format(scene.loaded_iter))
    test_dir = os.path.join(args.model_path, 'test', "ours_{}".format(scene.loaded_iter))
    gaussExtractor = GaussianExtractor(gaussians, render, pipe, bg_color=bg_color)  # 创建高斯提取器，用于渲染和网格提取
    
    # 如果不跳过训练集渲染，则渲染训练图像
    if not args.skip_train:
        print("export training images ...")
        os.makedirs(train_dir, exist_ok=True)  # 创建训练输出目录
        gaussExtractor.reconstruction(scene.getTrainCameras())  # 使用训练相机进行重建
        gaussExtractor.export_image(train_dir)  # 将渲染结果导出为图像
        
    
    # 如果不跳过测试集渲染且有测试相机，则渲染测试图像
    if (not args.skip_test) and (len(scene.getTestCameras()) > 0):
        print("export rendered testing images ...")
        os.makedirs(test_dir, exist_ok=True)  # 创建测试输出目录
        gaussExtractor.reconstruction(scene.getTestCameras())  # 使用测试相机进行重建
        gaussExtractor.export_image(test_dir)  # 将渲染结果导出为图像
    
    
    # 如果需要渲染路径，则生成相机轨迹并渲染视频
    if args.render_path:
        print("render videos ...")
        traj_dir = os.path.join(args.model_path, 'traj', "ours_{}".format(scene.loaded_iter))
        os.makedirs(traj_dir, exist_ok=True)  # 创建轨迹输出目录
        n_fames = 240  # 设置帧数
        cam_traj = generate_path(scene.getTrainCameras(), n_frames=n_fames)  # 基于训练相机生成相机轨迹
        gaussExtractor.reconstruction(cam_traj)  # 使用生成的相机轨迹进行重建
        gaussExtractor.export_image(traj_dir)  # 将渲染结果导出为图像
        create_videos(base_dir=traj_dir,
                    input_dir=traj_dir, 
                    out_name='render_traj', 
                    num_frames=n_fames)  # 将渲染的帧创建为视频

    # 如果不跳过网格导出，则导出3D网格
    if not args.skip_mesh:
        print("export mesh ...")
        os.makedirs(train_dir, exist_ok=True)  # 确保输出目录存在
        # 将活动球谐函数度数设为0，仅导出漫反射纹理
        gaussExtractor.gaussians.active_sh_degree = 0
        gaussExtractor.reconstruction(scene.getTrainCameras())  # 使用训练相机进行重建
        # 提取网格并保存
        if args.unbounded:
            # 使用无边界模式提取网格
            name = 'fuse_unbounded.ply'
            mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
        else:
            # 使用有边界模式提取网格
            name = 'fuse.ply'
            # 如果未指定参数，则计算合适的默认值
            depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0  else args.depth_trunc
            voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
            sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
            mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
        
        # 保存原始网格
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name), mesh)
        print("mesh saved at {}".format(os.path.join(train_dir, name)))
        # 对网格进行后处理并保存，保留最大的N个集群
        mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
        o3d.io.write_triangle_mesh(os.path.join(train_dir, name.replace('.ply', '_post.ply')), mesh_post)
        print("mesh post processed saved at {}".format(os.path.join(train_dir, name.replace('.ply', '_post.ply'))))