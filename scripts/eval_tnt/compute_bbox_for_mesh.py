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

# 此脚本用于计算网格的边界框（bounding box）
# 需要Open3D的Python绑定，请在运行此脚本前按照setup.py中的指示进行设置
import numpy as np
import open3d as o3d  # 导入Open3D库用于3D数据处理
import os
import argparse
import torch

from config import scenes_tau_dict  # 导入场景阈值字典
from registration import (
    trajectory_alignment,  # 轨迹对齐函数
    registration_vol_ds,   # 体素下采样配准函数
    registration_unif,     # 均匀下采样配准函数
    read_trajectory,       # 读取轨迹函数
)
from help_func import auto_orient_and_center_poses  # 自动定向和中心化相机姿态
from trajectory_io import CameraPose  # 相机姿态数据结构
from evaluation import EvaluateHisto  # 评估直方图函数
from util import make_dir  # 创建目录函数
from plot import plot_graph  # 绘图函数


def run_evaluation(dataset_dir, traj_path, ply_path, out_dir, view_crop):
    """
    运行评估函数
    
    参数:
    dataset_dir: 数据集目录路径
    traj_path: 相机轨迹文件路径
    ply_path: 重建点云文件路径
    out_dir: 输出目录
    view_crop: 是否可视化裁剪点云
    """
    scene = os.path.basename(os.path.normpath(dataset_dir))  # 获取场景名称

    if scene not in scenes_tau_dict:
        print(dataset_dir, scene)
        raise Exception("invalid dataset-dir, not in scenes_tau_dict")  # 如果场景不在预定义字典中则报错

    print("")
    print("===========================")
    print("Evaluating %s" % scene)  # 打印正在评估的场景名称
    print("===========================")

    dTau = scenes_tau_dict[scene]  # 获取该场景的阈值
    # 将裁剪文件、GT文件、COLMAP SfM日志文件和相应场景的对齐放在数据集目录的同名文件夹中
    colmap_ref_logfile = os.path.join(dataset_dir, scene + "_COLMAP_SfM.log")  # COLMAP SfM日志文件路径

    # 用于真实点云的对齐文件
    alignment = os.path.join(dataset_dir, scene + "_trans.txt")  # 对齐变换矩阵文件
    gt_filen = os.path.join(dataset_dir, scene + ".ply")  # 真实点云文件
    # 这个裁剪文件也是关于真实点云的，我们可以使用它
    # 否则我们需要自己裁剪估计的点云
    cropfile = os.path.join(dataset_dir, scene + ".json")  # 裁剪文件路径
    # 这个不是很必要
    map_file = os.path.join(dataset_dir, scene + "_mapping_reference.txt")  # 映射参考文件
    if not os.path.isfile(map_file):
        map_file = None
    map_file = None  # 设置为None表示不使用映射文件

    make_dir(out_dir)  # 创建输出目录

    # 加载重建结果和相应的真实数据(GT)
    print(ply_path)
    pcd = o3d.io.read_point_cloud(ply_path)  # 读取重建点云
    print(gt_filen)
    gt_pcd = o3d.io.read_point_cloud(gt_filen)  # 读取真实点云

    gt_trans = np.loadtxt(alignment)  # 加载对齐变换矩阵
    print(traj_path)
    traj_to_register = []  # 待注册的轨迹
    if traj_path.endswith('.npy'):  # 如果是npy格式的轨迹文件
        ld = np.load(traj_path)
        for i in range(len(ld)):
            traj_to_register.append(CameraPose(meta=None, mat=ld[i]))
    elif traj_path.endswith('.json'):  # 如果是json格式的轨迹文件(instant-npg或sdfstudio格式)
        import json
        with open(traj_path, encoding='UTF-8') as f:
            meta = json.load(f)
        poses_dict = {}
        for i, frame in enumerate(meta['frames']):
            filepath = frame['file_path']
            new_i = int(filepath[13:18]) - 1
            poses_dict[new_i] = np.array(frame['transform_matrix'])
        poses = []
        for i in range(len(poses_dict)):
            poses.append(poses_dict[i])
        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        # 自动定向和中心化相机姿态
        poses, _ = auto_orient_and_center_poses(poses, method='up', center_poses=True)
        # 缩放姿态
        scale_factor = 1.0 / float(torch.max(torch.abs(poses[:, :3, 3])))
        poses[:, :3, 3] *= scale_factor
        poses = poses.numpy()
        for i in range(len(poses)):
            traj_to_register.append(CameraPose(meta=None, mat=poses[i]))

    else:  # 其他格式的轨迹文件
        traj_to_register = read_trajectory(traj_path)
    print(colmap_ref_logfile)
    gt_traj_col = read_trajectory(colmap_ref_logfile)  # 读取COLMAP参考轨迹

    # 进行轨迹对齐
    trajectory_transform = trajectory_alignment(map_file, traj_to_register,
                                                gt_traj_col, gt_trans, scene)
    # 计算逆变换
    inv_transform = np.linalg.inv(trajectory_transform)
    points = np.asarray(gt_pcd.points)
    # 应用变换到点云
    points = points @ inv_transform[:3, :3].T + inv_transform[:3, 3:].T
    # 打印点云的最小和最大坐标
    print(points.min(axis=0), points.max(axis=0))
    # 打印点云的边界框坐标（以列表形式）
    print(np.concatenate([points.min(axis=0), points.max(axis=0)]).reshape(-1).tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="path to a dataset/scene directory containing X.json, X.ply, ...",  # 数据集目录路径
    )
    parser.add_argument(
        "--traj-path",
        type=str,
        required=True,
        help=
        "path to trajectory file. See `convert_to_logfile.py` to create this file.",  # 轨迹文件路径
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
    args = parser.parse_args()  # 解析命令行参数

    args.view_crop = False #  (args.view_crop > 0)  # 设置是否查看裁剪点云
    if args.out_dir.strip() == "":  # 如果未指定输出目录
        args.out_dir = os.path.join(os.path.dirname(args.ply_path),
                                    "evaluation")  # 在ply文件所在目录创建evaluation子目录

    # 运行评估函数
    run_evaluation(
        dataset_dir=args.dataset_dir,
        traj_path=args.traj_path,
        ply_path=args.ply_path,
        out_dir=args.out_dir,
        view_crop=args.view_crop
    )
