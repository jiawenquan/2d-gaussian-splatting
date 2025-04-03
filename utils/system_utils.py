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

from errno import EEXIST
from os import makedirs, path
import os

# 创建目录，相当于命令行中的mkdir -p
def mkdir_p(folder_path):
    """
    创建目录。等效于在命令行使用mkdir -p
    如果目录已经存在，则不会引发错误
    """
    try:
        # 尝试创建目录
        makedirs(folder_path)
    except OSError as exc: # Python >2.5
        # 如果目录已存在，忽略错误
        if exc.errno == EEXIST and path.isdir(folder_path):
            pass
        else:
            # 其他错误则继续抛出
            raise

# 在指定文件夹中搜索最大迭代次数，用于恢复训练
def searchForMaxIteration(folder):
    """
    在指定文件夹中搜索最大迭代次数
    通过解析文件名来确定保存的最大迭代次数
    """
    # 从文件名中提取迭代次数，并找出最大值
    saved_iters = [int(fname.split("_")[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)
