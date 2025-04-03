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
import json


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Arguments to evaluate the mesh.'  # 用于评估网格的参数
    )

    parser.add_argument('--input_mesh', type=str,  help='path to the mesh to be evaluated')  # 要评估的网格路径
    parser.add_argument('--scene', type=str,  help='scan id of the input mesh')  # 输入网格的扫描ID
    parser.add_argument('--output_dir', type=str, default='evaluation_results_single', help='path to the output folder')  # 输出文件夹路径
    parser.add_argument('--TNT', type=str,  default='Offical_DTU_Dataset', help='path to the GT DTU point clouds')  # GT DTU点云的路径
    args = parser.parse_args()  # 解析命令行参数


    TNT_Dataset = args.TNT  # 获取数据集路径
    out_dir = args.output_dir  # 获取输出目录
    Path(out_dir).mkdir(parents=True, exist_ok=True)  # 创建输出目录(如果不存在)
    scene = args.scene  # 获取场景ID
    ply_file = args.input_mesh  # 获取输入网格文件
    result_mesh_file = os.path.join(out_dir, "culled_mesh.ply")  # 设置结果网格文件路径
    # read scene.json
    # 这里有一个未完成的命令字符串，可能是用于执行评估的命令
    f"python run.py --dataset-dir {ply_file} --traj-path {TNT_Dataset}/{scene}/{scene}_COLMAP_SfM.log --ply-path {TNT_Dataset}/{scene}/{scene}_COLMAP.ply"