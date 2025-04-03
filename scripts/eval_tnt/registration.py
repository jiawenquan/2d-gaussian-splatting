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

from trajectory_io import read_trajectory, convert_trajectory_to_pointcloud
import copy
import numpy as np
import open3d as o3d

MAX_POINT_NUMBER = 4e6  # 最大点数限制


def read_mapping(filename):
    """
    从映射文件读取采样帧映射关系
    
    参数:
    filename: 映射文件路径
    
    返回:
    包含采样帧数量、总帧数和映射关系的列表
    """
    mapping = []
    with open(filename, "r") as f:
        n_sampled_frames = int(f.readline())  # 采样帧数量
        n_total_frames = int(f.readline())  # 总帧数
        mapping = np.zeros(shape=(n_sampled_frames, 2))  # 创建映射数组
        metastr = f.readline()
        for iter in range(n_sampled_frames):
            metadata = list(map(int, metastr.split()))
            mapping[iter, :] = metadata
            metastr = f.readline()
    return [n_sampled_frames, n_total_frames, mapping]


def gen_sparse_trajectory(mapping, f_trajectory):
    """
    根据映射生成稀疏轨迹
    
    参数:
    mapping: 帧映射关系
    f_trajectory: 完整轨迹
    
    返回:
    稀疏轨迹
    """
    sparse_traj = []
    for m in mapping:
        sparse_traj.append(f_trajectory[int(m[1] - 1)])
    return sparse_traj


def trajectory_alignment(map_file, traj_to_register, gt_traj_col, gt_trans,
                         scene):
    """
    轨迹对齐函数
    
    参数:
    map_file: 映射文件路径
    traj_to_register: 待注册的轨迹
    gt_traj_col: 真实轨迹
    gt_trans: 真实变换矩阵
    scene: 场景名称
    
    返回:
    对齐变换矩阵
    """
    traj_pcd_col = convert_trajectory_to_pointcloud(gt_traj_col)  # 将真实轨迹转换为点云
    if gt_trans is not None:
        traj_pcd_col.transform(gt_trans)  # 应用变换矩阵
    # 创建对应关系，每个点与自身对应
    corres = o3d.utility.Vector2iVector(
        np.asarray(list(map(lambda x: [x, x], range(len(gt_traj_col))))))
    # 设置RANSAC收敛标准
    rr = o3d.registration.RANSACConvergenceCriteria()
    rr.max_iteration = 100000
    rr.max_validation = 100000

    # 如果轨迹过长并且存在映射文件，则使用稀疏轨迹
    # 这种情况下使用的是包含每个电影帧的日志文件（详见教程）
    if len(traj_to_register) > 1600 and map_file is not None:
        n_sampled_frames, n_total_frames, mapping = read_mapping(map_file)
        traj_col2 = gen_sparse_trajectory(mapping, traj_to_register)
        traj_to_register_pcd = convert_trajectory_to_pointcloud(traj_col2)
    else:
        print("Estimated trajectory will leave as it is, no sparsity op is performed!")
        traj_to_register_pcd = convert_trajectory_to_pointcloud(
            traj_to_register)
    # 随机变量，用于添加随机扰动
    randomvar = 0.0
    if randomvar < 1e-5:
        traj_to_register_pcd_rand = traj_to_register_pcd
    else:
        nr_of_cam_pos = len(traj_to_register_pcd.points)
        # 添加随机扰动
        rand_number_added = np.asanyarray(traj_to_register_pcd.points) * (
            np.random.rand(nr_of_cam_pos, 3) * randomvar - randomvar / 2.0 + 1)
        list_rand = list(rand_number_added)
        traj_to_register_pcd_rand = o3d.geometry.PointCloud()
        for elem in list_rand:
            traj_to_register_pcd_rand.points.append(elem)

    # 基于对齐的COLMAP SfM数据进行粗配准
    reg = o3d.registration.registration_ransac_based_on_correspondence(
        traj_to_register_pcd_rand,
        traj_pcd_col,
        corres,
        0.2,
        o3d.registration.TransformationEstimationPointToPoint(True),
        6,
        rr,
    )
    return reg.transformation  # 返回配准变换矩阵


def crop_and_downsample(
        pcd,
        crop_volume,
        down_sample_method="voxel",
        voxel_size=0.01,
        trans=np.identity(4),
):
    """
    裁剪和下采样点云
    
    参数:
    pcd: 输入点云
    crop_volume: 裁剪体积
    down_sample_method: 下采样方法，"voxel"或"uniform"
    voxel_size: 体素大小
    trans: 变换矩阵
    
    返回:
    裁剪和下采样后的点云
    """
    pcd_copy = copy.deepcopy(pcd)  # 深拷贝点云
    pcd_copy.transform(trans)  # 应用变换
    pcd_crop = crop_volume.crop_point_cloud(pcd_copy)  # 裁剪点云
    if down_sample_method == "voxel":
        # return voxel_down_sample(pcd_crop, voxel_size)
        return pcd_crop.voxel_down_sample(voxel_size)  # 体素下采样
    elif down_sample_method == "uniform":
        n_points = len(pcd_crop.points)
        # 如果点数超过最大限制，则进行均匀下采样
        if n_points > MAX_POINT_NUMBER:
            ds_rate = int(round(n_points / float(MAX_POINT_NUMBER)))
            return pcd_crop.uniform_down_sample(ds_rate)  # 均匀下采样
    return pcd_crop  # 返回裁剪后的点云


def registration_unif(
    source,
    gt_target,
    init_trans,
    crop_volume,
    threshold,
    max_itr,
    max_size=4 * MAX_POINT_NUMBER,
    verbose=True,
):
    """
    基于均匀下采样的点云配准
    
    参数:
    source: 源点云
    gt_target: 目标点云（真实点云）
    init_trans: 初始变换矩阵
    crop_volume: 裁剪体积
    threshold: 配准阈值
    max_itr: 最大迭代次数
    max_size: 最大点数
    verbose: 是否输出详细信息
    
    返回:
    配准结果
    """
    if verbose:
        print("[Registration] threshold: %f" % threshold)
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    # 裁剪和均匀下采样源点云
    s = crop_and_downsample(source,
                            crop_volume,
                            down_sample_method="uniform",
                            trans=init_trans)
    # 裁剪和均匀下采样目标点云
    t = crop_and_downsample(gt_target,
                            crop_volume,
                            down_sample_method="uniform")
    # 使用ICP算法进行点云配准
    reg = o3d.registration.registration_icp(
        s,
        t,
        threshold,
        np.identity(4),
        o3d.registration.TransformationEstimationPointToPoint(True),
        o3d.registration.ICPConvergenceCriteria(1e-6, max_itr),
    )
    # 将配准变换与初始变换相乘
    reg.transformation = np.matmul(reg.transformation, init_trans)
    return reg  # 返回配准结果


def registration_vol_ds(
    source,
    gt_target,
    init_trans,
    crop_volume,
    voxel_size,
    threshold,
    max_itr,
    verbose=True,
):
    """
    基于体素下采样的点云配准
    
    参数:
    source: 源点云
    gt_target: 目标点云（真实点云）
    init_trans: 初始变换矩阵
    crop_volume: 裁剪体积
    voxel_size: 体素大小
    threshold: 配准阈值
    max_itr: 最大迭代次数
    verbose: 是否输出详细信息
    
    返回:
    配准结果
    """
    if verbose:
        print("[Registration] voxel_size: %f, threshold: %f" %
              (voxel_size, threshold))
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    # 裁剪和体素下采样源点云
    s = crop_and_downsample(
        source,
        crop_volume,
        down_sample_method="voxel",
        voxel_size=voxel_size,
        trans=init_trans,
    )
    # 裁剪和体素下采样目标点云
    t = crop_and_downsample(
        gt_target,
        crop_volume,
        down_sample_method="voxel",
        voxel_size=voxel_size,
    )
    # 使用ICP算法进行点云配准
    reg = o3d.registration.registration_icp(
        s,
        t,
        threshold,
        np.identity(4),
        o3d.registration.TransformationEstimationPointToPoint(True),
        o3d.registration.ICPConvergenceCriteria(1e-6, max_itr),
    )
    # 将配准变换与初始变换相乘
    reg.transformation = np.matmul(reg.transformation, init_trans)
    return reg  # 返回配准结果
