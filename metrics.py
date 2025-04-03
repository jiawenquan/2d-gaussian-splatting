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

# 导入必要的库
from pathlib import Path  # 用于处理文件路径
import os  # 操作系统接口
from PIL import Image  # 图像处理库
import torch  # PyTorch深度学习框架
import torchvision.transforms.functional as tf  # PyTorch图像转换函数
from utils.loss_utils import ssim  # 结构相似性指标
from lpipsPyTorch import lpips  # 学习感知图像块相似性指标
import json  # JSON数据处理
from tqdm import tqdm  # 进度条显示
from utils.image_utils import psnr  # 峰值信噪比指标
from argparse import ArgumentParser  # 命令行参数解析器

def readImages(renders_dir, gt_dir):
    """
    读取渲染图像和对应的真实图像(ground truth)
    
    参数:
        renders_dir: 渲染图像目录路径
        gt_dir: 真实图像目录路径
        
    返回:
        renders: 渲染图像列表(张量形式)
        gts: 真实图像列表(张量形式)
        image_names: 图像文件名列表
    """
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)  # 打开渲染图像
        gt = Image.open(gt_dir / fname)  # 打开对应的真实图像
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())  # 转换为张量并移至GPU
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())  # 转换为张量并移至GPU
        image_names.append(fname)
    return renders, gts, image_names

def evaluate(model_paths):
    """
    评估模型的渲染质量
    
    参数:
        model_paths: 模型路径列表
    """
    # 初始化结果存储字典
    full_dict = {}  # 存储每个场景每种方法的平均指标
    per_view_dict = {}  # 存储每个场景每种方法每个视角的指标
    full_dict_polytopeonly = {}  # （未使用）多面体相关结果
    per_view_dict_polytopeonly = {}  # （未使用）多面体相关结果

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)
            # 为每个场景初始化结果字典
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            test_dir = Path(scene_dir) / "test"  # 测试目录路径

            for method in os.listdir(test_dir):
                print("Method:", method)
                
                # 为每种方法初始化结果字典
                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir/ "gt"  # 真实图像目录
                renders_dir = method_dir / "renders"  # 渲染图像目录
                renders, gts, image_names = readImages(renders_dir, gt_dir)  # 读取图像

                # 存储每个图像的评估指标
                ssims = []  # 结构相似性指标列表
                psnrs = []  # 峰值信噪比指标列表
                lpipss = []  # 感知相似性指标列表

                # 对每张图像计算评估指标
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))  # 计算SSIM
                    psnrs.append(psnr(renders[idx], gts[idx]))  # 计算PSNR
                    lpipss.append(lpips(renders[idx], gts[idx], net_type='vgg'))  # 计算LPIPS

                # 打印平均评估指标
                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("  PSNR : {:>12.7f}".format(torch.tensor(psnrs).mean(), ".5"))
                print("  LPIPS: {:>12.7f}".format(torch.tensor(lpipss).mean(), ".5"))
                print("")

                # 更新结果字典，存储平均指标
                full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                        "PSNR": torch.tensor(psnrs).mean().item(),
                                                        "LPIPS": torch.tensor(lpipss).mean().item()})
                # 更新结果字典，存储每个视角的详细指标
                per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                            "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                            "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)}})

            # 将结果保存为JSON文件
            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)  # 保存平均指标
            with open(scene_dir + "/per_view.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)  # 保存每个视角的详细指标
        except:
            print("Unable to compute metrics for model", scene_dir)  # 处理评估过程中的异常

if __name__ == "__main__":
    # 设置GPU设备
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # 设置命令行参数解析器
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])  # 添加模型路径参数
    args = parser.parse_args()  # 解析命令行参数
    evaluate(args.model_paths)  # 评估指定模型 