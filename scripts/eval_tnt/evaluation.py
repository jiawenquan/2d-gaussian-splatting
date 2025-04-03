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

import json
import copy
import os
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def read_alignment_transformation(filename):
    """
    从文件中读取对齐变换矩阵
    
    参数:
    filename: 包含对齐变换的JSON文件路径
    
    返回:
    4x4的变换矩阵
    """
    with open(filename) as data_file:
        data = json.load(data_file)
    return np.asarray(data["transformation"]).reshape((4, 4)).transpose()


def write_color_distances(path, pcd, distances, max_distance):
    """
    将距离信息作为颜色写入点云并保存
    
    参数:
    path: 输出点云文件路径
    pcd: 点云对象
    distances: 每个点的距离值
    max_distance: 最大距离值，用于颜色归一化
    """
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    # cmap = plt.get_cmap("afmhot")
    cmap = plt.get_cmap("hot_r")  # 使用热度图颜色映射
    distances = np.array(distances)
    # 将距离值映射到[0, max_distance]范围内，并转换为颜色
    colors = cmap(np.minimum(distances, max_distance) / max_distance)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)  # 设置点云颜色
    o3d.io.write_point_cloud(path, pcd)  # 保存点云


def EvaluateHisto(
    source,
    target,
    trans,
    crop_volume,
    voxel_size,
    threshold,
    filename_mvs,
    plot_stretch,
    scene_name,
    view_crop,
    verbose=True,
):
    """
    评估源点云与目标点云之间的匹配质量，计算精度、召回率和F-score
    
    参数:
    source: 源点云（估计点云）
    target: 目标点云（真实点云）
    trans: 对齐变换矩阵
    crop_volume: 用于裁剪点云的体积
    voxel_size: 体素大小，用于下采样
    threshold: 距离阈值，用于计算精度和召回率
    filename_mvs: 输出文件路径前缀
    plot_stretch: 绘图拉伸因子
    scene_name: 场景名称
    view_crop: 是否可视化裁剪结果
    verbose: 是否打印详细信息
    
    返回:
    包含精度、召回率、F-score和绘图数据的列表
    """
    print("[EvaluateHisto]")
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    s = copy.deepcopy(source)  # 深拷贝源点云
    s.transform(trans)  # 应用变换矩阵
    if crop_volume is not None:
        s = crop_volume.crop_point_cloud(s)  # 裁剪点云
        if view_crop:
            o3d.visualization.draw_geometries([s, ])  # 可视化裁剪结果
    else:
        print("No bounding box provided to crop estimated point cloud, leaving it as the loaded version!!")
    s = s.voxel_down_sample(voxel_size)  # 体素下采样
    s.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))  # 估计法线
    print(filename_mvs + "/" + scene_name + ".precision.ply")  # 打印精度文件路径

    t = copy.deepcopy(target)  # 深拷贝目标点云
    if crop_volume is not None:
        t = crop_volume.crop_point_cloud(t)  # 裁剪点云
    else:
        print("No bounding box provided to crop groundtruth point cloud, leaving it as the loaded version!!")

    t = t.voxel_down_sample(voxel_size)  # 体素下采样
    t.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))  # 估计法线
    print("[compute_point_cloud_to_point_cloud_distance]")
    distance1 = s.compute_point_cloud_distance(t)  # 计算源点云到目标点云的距离
    print("[compute_point_cloud_to_point_cloud_distance]")
    distance2 = t.compute_point_cloud_distance(s)  # 计算目标点云到源点云的距离

    # write the distances to bin files
    # np.array(distance1).astype("float64").tofile(
    #     filename_mvs + "/" + scene_name + ".precision.bin"
    # )
    # np.array(distance2).astype("float64").tofile(
    #     filename_mvs + "/" + scene_name + ".recall.bin"
    # )

    # Colorize the poincloud files prith the precision and recall values
    # o3d.io.write_point_cloud(
    #     filename_mvs + "/" + scene_name + ".precision.ply", s
    # )
    # o3d.io.write_point_cloud(
    #     filename_mvs + "/" + scene_name + ".precision.ncb.ply", s
    # )
    # o3d.io.write_point_cloud(filename_mvs + "/" + scene_name + ".recall.ply", t)

    source_n_fn = filename_mvs + "/" + scene_name + ".precision.ply"  # 精度点云输出路径
    target_n_fn = filename_mvs + "/" + scene_name + ".recall.ply"  # 召回率点云输出路径

    print("[ViewDistances] Add color coding to visualize error")
    # eval_str_viewDT = (
    #     OPEN3D_EXPERIMENTAL_BIN_PATH
    #     + "ViewDistances "
    #     + source_n_fn
    #     + " --max_distance "
    #     + str(threshold * 3)
    #     + " --write_color_back --without_gui"
    # )
    # os.system(eval_str_viewDT)
    write_color_distances(source_n_fn, s, distance1, 3 * threshold)  # 将距离信息以颜色形式写入精度点云

    print("[ViewDistances] Add color coding to visualize error")
    # eval_str_viewDT = (
    #     OPEN3D_EXPERIMENTAL_BIN_PATH
    #     + "ViewDistances "
    #     + target_n_fn
    #     + " --max_distance "
    #     + str(threshold * 3)
    #     + " --write_color_back --without_gui"
    # )
    # os.system(eval_str_viewDT)
    write_color_distances(target_n_fn, t, distance2, 3 * threshold)  # 将距离信息以颜色形式写入召回率点云

    # 获取直方图和F-score
    [
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
    ] = get_f1_score_histo2(threshold, filename_mvs, plot_stretch, distance1,
                            distance2)
    # 保存评估结果
    np.savetxt(filename_mvs + "/" + scene_name + ".recall.txt", cum_target)  # 保存召回率数据
    np.savetxt(filename_mvs + "/" + scene_name + ".precision.txt", cum_source)  # 保存精度数据
    np.savetxt(
        filename_mvs + "/" + scene_name + ".prf_tau_plotstr.txt",
        np.array([precision, recall, fscore, threshold, plot_stretch]),  # 保存精度、召回率、F-score等信息
    )

    return [
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
    ]


def get_f1_score_histo2(threshold,
                        filename_mvs,
                        plot_stretch,
                        distance1,
                        distance2,
                        verbose=True):
    """
    根据点云间距离计算F1分数和直方图数据
    
    参数:
    threshold: 距离阈值
    filename_mvs: 输出文件路径前缀
    plot_stretch: 绘图拉伸因子
    distance1: 源点云到目标点云的距离
    distance2: 目标点云到源点云的距离
    verbose: 是否打印详细信息
    
    返回:
    精度、召回率、F-score和用于绘图的数据
    """
    print("[get_f1_score_histo2]")
    dist_threshold = threshold
    if len(distance1) and len(distance2):
        # 计算召回率：距离小于阈值的target点比例
        recall = float(sum(d < threshold for d in distance2)) / float(
            len(distance2))
        # 计算精度：距离小于阈值的source点比例
        precision = float(sum(d < threshold for d in distance1)) / float(
            len(distance1))
        # 计算F-score：精度和召回率的调和平均
        fscore = 2 * recall * precision / (recall + precision)
        
        # 为源点云计算距离累积直方图
        num = len(distance1)
        bins = np.arange(0, dist_threshold * plot_stretch, dist_threshold / 100)
        hist, edges_source = np.histogram(distance1, bins)
        cum_source = np.cumsum(hist).astype(float) / num

        # 为目标点云计算距离累积直方图
        num = len(distance2)
        bins = np.arange(0, dist_threshold * plot_stretch, dist_threshold / 100)
        hist, edges_target = np.histogram(distance2, bins)
        cum_target = np.cumsum(hist).astype(float) / num

    else:
        # 如果距离为空，则所有指标都为0
        precision = 0
        recall = 0
        fscore = 0
        edges_source = np.array([0])
        cum_source = np.array([0])
        edges_target = np.array([0])
        cum_target = np.array([0])

    return [
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
    ]
