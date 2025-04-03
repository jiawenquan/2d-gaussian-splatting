# ----------------------------------------------------------------------------
# -                   TanksAndTemples Website Toolbox                        -
# -                    http://www.tanksandtemples.org                        -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2017
# Arno Knapitsch <arno.knapitsch@gmail.com >
# Jaesik Park <syncle@gmail.com>
# Qian-Yi Zhou <Qianyi.Zhou@gmail.com>
# Vladlen Koltun <vkoltun@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ----------------------------------------------------------------------------
#
# This python script is for downloading dataset from www.tanksandtemples.org
# The dataset has a different license, please refer to
# https://tanksandtemples.org/license/

# this script requires Open3D python binding
# please follow the intructions in setup.py before running this script.
import numpy as np
import open3d as o3d
import os
import argparse
# import torch

from config import scenes_tau_dict
from registration import (
    trajectory_alignment,
    registration_vol_ds,
    registration_unif,
    read_trajectory,
)
# from help_func import auto_orient_and_center_poses
from trajectory_io import CameraPose
from evaluation import EvaluateHisto
from util import make_dir
from plot import plot_graph


def run_evaluation(dataset_dir, traj_path, ply_path, out_dir, view_crop):
    """
    对重建的点云进行评估，与真实数据进行比较
    
    参数:
        dataset_dir: 数据集目录路径
        traj_path: 相机轨迹文件路径
        ply_path: 重建点云文件路径
        out_dir: 输出目录
        view_crop: 是否查看裁剪后的点云
    """
    # 获取场景名称
    scene = os.path.basename(os.path.normpath(dataset_dir))

    # 检查场景是否在预定义的字典中
    if scene not in scenes_tau_dict:
        print(dataset_dir, scene)
        raise Exception("invalid dataset-dir, not in scenes_tau_dict")

    print("")
    print("===========================")
    print("Evaluating %s" % scene)
    print("===========================")

    # 获取该场景的距离阈值tau
    dTau = scenes_tau_dict[scene]
    # 设置各种文件路径
    # COLMAP SfM日志文件路径（包含相机位姿）
    colmap_ref_logfile = os.path.join(dataset_dir, scene + "_COLMAP_SfM.log")

    # 地面真实点云的变换矩阵路径
    alignment = os.path.join(dataset_dir, scene + "_trans.txt")
    # 地面真实点云路径
    gt_filen = os.path.join(dataset_dir, scene + ".ply")
    # 裁剪文件路径，用于裁剪点云
    cropfile = os.path.join(dataset_dir, scene + ".json")
    # 映射参考文件路径（可选）
    map_file = os.path.join(dataset_dir, scene + "_mapping_reference.txt")
    if not os.path.isfile(map_file):
        map_file = None
    map_file = None

    # 创建输出目录
    make_dir(out_dir)

    # 加载重建点云和真实点云
    print(ply_path)
    pcd = o3d.io.read_point_cloud(ply_path)
    # 添加面的中心点，增强点云密度
    import trimesh
    mesh = trimesh.load_mesh(ply_path)
    # 计算每个三角面的中心点
    sampled_vertices = mesh.vertices[mesh.faces].mean(axis=1)
    # 可选：可以添加基于面顶点的更多采样点
    # 将原始顶点和采样点合并
    vertices = np.concatenate([mesh.vertices, sampled_vertices], axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    
    # 加载地面真实点云
    print(gt_filen)
    gt_pcd = o3d.io.read_point_cloud(gt_filen)

    # 加载地面真实点云的变换矩阵
    gt_trans = np.loadtxt(alignment)
    
    # 加载相机轨迹
    print(traj_path)
    traj_to_register = []
    if traj_path.endswith('.npy'):
        # 如果是npy格式的轨迹文件
        ld = np.load(traj_path)
        for i in range(len(ld)):
            traj_to_register.append(CameraPose(meta=None, mat=ld[i]))
    elif traj_path.endswith('.json'): # instant-npg or sdfstudio format
        # 如果是json格式的轨迹文件（instant-NGP或sdfstudio格式）
        import json
        with open(traj_path, encoding='UTF-8') as f:
            meta = json.load(f)
        poses_dict = {}
        # 解析帧信息
        for i, frame in enumerate(meta['frames']):
            filepath = frame['file_path']
            new_i = int(filepath[13:18]) - 1
            poses_dict[new_i] = np.array(frame['transform_matrix'])
        poses = []
        for i in range(len(poses_dict)):
            poses.append(poses_dict[i])
        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        # 自动定向和中心化相机位姿
        poses, _ = auto_orient_and_center_poses(poses, method='up', center_poses=True)
        # 根据最大位移进行缩放
        scale_factor = 1.0 / float(torch.max(torch.abs(poses[:, :3, 3])))
        poses[:, :3, 3] *= scale_factor
        poses = poses.numpy()
        for i in range(len(poses)):
            traj_to_register.append(CameraPose(meta=None, mat=poses[i]))
    else:
        # 其他格式（默认COLMAP格式）
        traj_to_register = read_trajectory(traj_path)
    
    # 加载参考相机轨迹
    print(colmap_ref_logfile)
    gt_traj_col = read_trajectory(colmap_ref_logfile)

    # 进行轨迹对齐，将重建的相机轨迹与地面真实轨迹对齐
    trajectory_transform = trajectory_alignment(map_file, traj_to_register,
                                                gt_traj_col, gt_trans, scene)

    
    # 设置距离阈值
    dist_threshold = dTau
    # 加载裁剪体积
    vol = o3d.visualization.read_selection_polygon_volume(cropfile)
    
    # 进行三次迭代的配准优化
    # 第一次：使用较大的体素尺寸和距离阈值
    r2 = registration_vol_ds(pcd, gt_pcd, trajectory_transform, vol, dTau,
                             dTau * 80, 20)
    # 第二次：使用中等的体素尺寸和距离阈值
    r3 = registration_vol_ds(pcd, gt_pcd, r2.transformation, vol, dTau / 2.0,
                             dTau * 20, 20)
    # 第三次：使用均匀采样和较小的距离阈值
    r = registration_unif(pcd, gt_pcd, r3.transformation, vol, 2 * dTau, 20)
    trajectory_transform = r.transformation
    
    # 计算直方图和精度/召回率/F1分数
    plot_stretch = 5  # 绘图拉伸系数
    [
        precision,  # 精度
        recall,     # 召回率
        fscore,     # F1分数
        edges_source,  # 源点云直方图边缘
        cum_source,    # 源点云累积直方图
        edges_target,  # 目标点云直方图边缘
        cum_target,    # 目标点云累积直方图
    ] = EvaluateHisto(
        pcd,        # 重建点云
        gt_pcd,     # 真实点云
        trajectory_transform,  # 变换矩阵
        vol,        # 裁剪体积
        dTau / 2.0, # 距离阈值的一半
        dTau,       # 距离阈值
        out_dir,    # 输出目录
        plot_stretch,  # 绘图拉伸系数
        scene,      # 场景名
        view_crop   # 是否查看裁剪后的点云
    )
    eva = [precision, recall, fscore]
    # 打印评估结果
    print("==============================")
    print("evaluation result : %s" % scene)
    print("==============================")
    print("distance tau : %.3f" % dTau)
    print("precision : %.4f" % eva[0])  # 精度
    print("recall : %.4f" % eva[1])     # 召回率
    print("f-score : %.4f" % eva[2])    # F1分数
    print("==============================")

    # 绘制评估图表
    plot_graph(
        scene,          # 场景名
        fscore,         # F1分数
        dist_threshold, # 距离阈值
        edges_source,   # 源点云直方图边缘
        cum_source,     # 源点云累积直方图
        edges_target,   # 目标点云直方图边缘
        cum_target,     # 目标点云累积直方图
        plot_stretch,   # 绘图拉伸系数
        out_dir,        # 输出目录
    )


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="path to a dataset/scene directory containing X.json, X.ply, ...",  # 包含场景数据的目录路径
    )
    parser.add_argument(
        "--traj-path",
        type=str,
        required=True,
        help=
        "path to trajectory file. See `convert_to_logfile.py` to create this file.",  # 相机轨迹文件路径
    )
    parser.add_argument(
        "--ply-path",
        type=str,
        required=True,
        help="path to reconstruction ply file",  # 重建点云文件路径
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help=
        "output directory, default: an evaluation directory is created in the directory of the ply file",  # 输出目录
    )
    parser.add_argument(
        "--view-crop",
        type=int,
        default=0,
        help="whether view the crop pointcloud after aligned",  # 是否查看对齐后的裁剪点云
    )
    args = parser.parse_args()

    # 设置view_crop参数
    args.view_crop = False #  (args.view_crop > 0)
    # 如果没有指定输出目录，则在ply文件所在目录下创建evaluation目录
    if args.out_dir.strip() == "":
        args.out_dir = os.path.join(os.path.dirname(args.ply_path),
                                    "evaluation")

    # 运行评估
    run_evaluation(
        dataset_dir=args.dataset_dir,
        traj_path=args.traj_path,
        ply_path=args.ply_path,
        out_dir=args.out_dir,
        view_crop=args.view_crop
    )
