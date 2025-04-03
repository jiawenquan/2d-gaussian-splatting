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
from argparse import ArgumentParser

# 定义MipNeRF360数据集中的室外场景列表
mipnerf360_outdoor_scenes = ["bicycle", "flowers", "garden", "stump", "treehill"]
# 定义MipNeRF360数据集中的室内场景列表
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
# 以下是其它数据集场景(当前被注释掉)
# tanks_and_temples_scenes = ["truck", "train"]
# deep_blending_scenes = ["drjohnson", "playroom"] 

# 设置命令行参数解析器
parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")  # 跳过训练阶段的标志
parser.add_argument("--skip_rendering", action="store_true")  # 跳过渲染阶段的标志
parser.add_argument("--skip_metrics", action="store_true")  # 跳过指标计算阶段的标志
parser.add_argument("--output_path", default="eval/mipnerf360")  # 输出路径，默认为eval/mipnerf360
args, _ = parser.parse_known_args()  # 解析已知参数，忽略未知参数

# 合并所有场景列表
all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(mipnerf360_indoor_scenes)
# all_scenes.extend(tanks_and_temples_scenes)
# all_scenes.extend(deep_blending_scenes)

# 如果不跳过训练或渲染，则需要数据集路径
if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--mipnerf360', "-m360", required=True, type=str)  # MipNeRF360数据集路径
    # parser.add_argument("--tanksandtemples", "-tat", required=True, type=str)
    # parser.add_argument("--deepblending", "-db", required=True, type=str)
    args = parser.parse_args()  # 重新解析所有参数

# 训练阶段
if not args.skip_training:
    common_args = " --quiet --eval --test_iterations -1"  # 定义所有场景共用的训练参数
    # 处理室外场景，指定images_4目录
    for scene in mipnerf360_outdoor_scenes:
        source = args.mipnerf360 + "/" + scene  # 数据源路径
        os.system("python train.py -s " + source + " -i images_4 -m " + args.output_path + "/" + scene + common_args)  # 执行训练命令
    # 处理室内场景，指定images_2目录
    for scene in mipnerf360_indoor_scenes:
        source = args.mipnerf360 + "/" + scene  # 数据源路径
        os.system("python train.py -s " + source + " -i images_2 -m " + args.output_path + "/" + scene + common_args)  # 执行训练命令
    # 下面是其它数据集场景的训练代码(当前被注释掉)
    # for scene in tanks_and_temples_scenes:
        # source = args.tanksandtemples + "/" + scene
        # os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
    # for scene in deep_blending_scenes:
        # source = args.deepblending + "/" + scene
        # os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)

# 渲染阶段
if not args.skip_rendering:
    all_sources = []  # 用于存储所有数据源路径的列表
    # 为每个场景构建数据源路径
    for scene in mipnerf360_outdoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    for scene in mipnerf360_indoor_scenes:
        all_sources.append(args.mipnerf360 + "/" + scene)
    # for scene in tanks_and_temples_scenes:
        # all_sources.append(args.tanksandtemples + "/" + scene)
    # for scene in deep_blending_scenes:
        # all_sources.append(args.deepblending + "/" + scene)

    common_args = " --quiet --eval --skip_train"  # 定义所有场景共用的渲染参数
    # 对每个场景执行渲染
    for scene, source in zip(all_scenes, all_sources):
        os.system("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)  # 执行渲染命令

# 评估指标计算阶段
if not args.skip_metrics:
    scenes_string = ""  # 用于构建场景路径字符串
    # 为每个场景构建路径并添加到字符串中
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "
    
    # 执行指标计算命令
    os.system("python metrics.py -m " + scenes_string)