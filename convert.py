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

import os                      # 用于操作系统功能，如目录创建和文件操作
import logging                 # 用于记录程序运行时的日志信息
from argparse import ArgumentParser  # 用于解析命令行参数
import shutil                  # 用于高级文件操作，如复制和移动文件

# 这个Python脚本基于MipNerF 360仓库中提供的shell转换脚本
# This Python script is based on the shell converter script provided in the MipNerF 360 repository.

# 创建命令行参数解析器
parser = ArgumentParser("Colmap converter")
parser.add_argument("--no_gpu", action='store_true')          # 标志参数，指示不使用GPU
parser.add_argument("--skip_matching", action='store_true')   # 标志参数，跳过特征匹配步骤
parser.add_argument("--source_path", "-s", required=True, type=str)  # 必需参数，指定源图像路径
parser.add_argument("--camera", default="OPENCV", type=str)   # 指定相机模型，默认为OPENCV
parser.add_argument("--colmap_executable", default="", type=str)  # 指定COLMAP可执行文件的路径
parser.add_argument("--resize", action="store_true")          # 标志参数，指示是否调整图像大小
parser.add_argument("--magick_executable", default="", type=str)  # 指定ImageMagick可执行文件的路径
args = parser.parse_args()  # 解析命令行参数

# 设置COLMAP命令，如果提供了路径则使用，否则使用默认的"colmap"命令
colmap_command = '"{}"'.format(args.colmap_executable) if len(args.colmap_executable) > 0 else "colmap"
# 设置ImageMagick命令，如果提供了路径则使用，否则使用默认的"magick"命令
magick_command = '"{}"'.format(args.magick_executable) if len(args.magick_executable) > 0 else "magick"
# 设置是否使用GPU：如果没有指定--no_gpu标志，则使用GPU(1)，否则不使用(0)
use_gpu = 1 if not args.no_gpu else 0

# 如果未指定跳过匹配步骤，执行COLMAP的完整处理流程
if not args.skip_matching:
    # 创建存储扭曲图像相关数据的目录
    os.makedirs(args.source_path + "/distorted/sparse", exist_ok=True)

    ## 特征提取步骤
    # 构建COLMAP特征提取命令，从输入图像中提取特征点
    feat_extracton_cmd = colmap_command + " feature_extractor "\
        "--database_path " + args.source_path + "/distorted/database.db \
        --image_path " + args.source_path + "/input \
        --ImageReader.single_camera 1 \
        --ImageReader.camera_model " + args.camera + " \
        --SiftExtraction.use_gpu " + str(use_gpu)
    # 执行特征提取命令
    exit_code = os.system(feat_extracton_cmd)
    # 如果命令执行失败，记录错误并退出
    if exit_code != 0:
        logging.error(f"Feature extraction failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ## 特征匹配步骤
    # 构建COLMAP特征匹配命令，匹配不同图像间的特征点
    feat_matching_cmd = colmap_command + " exhaustive_matcher \
        --database_path " + args.source_path + "/distorted/database.db \
        --SiftMatching.use_gpu " + str(use_gpu)
    # 执行特征匹配命令
    exit_code = os.system(feat_matching_cmd)
    # 如果命令执行失败，记录错误并退出
    if exit_code != 0:
        logging.error(f"Feature matching failed with code {exit_code}. Exiting.")
        exit(exit_code)

    ### 束调整步骤（Bundle Adjustment）
    # 默认的Mapper容差过大，减小它可以加速束调整步骤
    # The default Mapper tolerance is unnecessarily large,
    # decreasing it speeds up bundle adjustment steps.
    # 构建COLMAP mapper命令，进行3D重建和相机位姿估计
    mapper_cmd = (colmap_command + " mapper \
        --database_path " + args.source_path + "/distorted/database.db \
        --image_path "  + args.source_path + "/input \
        --output_path "  + args.source_path + "/distorted/sparse \
        --Mapper.ba_global_function_tolerance=0.000001")
    # 执行mapper命令
    exit_code = os.system(mapper_cmd)
    # 如果命令执行失败，记录错误并退出
    if exit_code != 0:
        logging.error(f"Mapper failed with code {exit_code}. Exiting.")
        exit(exit_code)

### 图像去畸变步骤
## 需要将图像去畸变为理想的针孔相机内参模型
# 构建COLMAP图像去畸变命令
img_undist_cmd = (colmap_command + " image_undistorter \
    --image_path " + args.source_path + "/input \
    --input_path " + args.source_path + "/distorted/sparse/0 \
    --output_path " + args.source_path + "\
    --output_type COLMAP")
# 执行图像去畸变命令
exit_code = os.system(img_undist_cmd)
# 如果命令执行失败，记录错误并退出
if exit_code != 0:
    logging.error(f"Mapper failed with code {exit_code}. Exiting.")
    exit(exit_code)

# 获取生成的sparse目录中的文件列表
files = os.listdir(args.source_path + "/sparse")
# 确保存在目标子目录
os.makedirs(args.source_path + "/sparse/0", exist_ok=True)
# 将文件从源目录移动到目标子目录
# Copy each file from the source directory to the destination directory
for file in files:
    # 跳过已经存在的'0'目录
    if file == '0':
        continue
    # 构建源文件和目标文件的完整路径
    source_file = os.path.join(args.source_path, "sparse", file)
    destination_file = os.path.join(args.source_path, "sparse", "0", file)
    # 移动文件
    shutil.move(source_file, destination_file)

# 如果指定了resize参数，创建不同分辨率的图像副本
if(args.resize):
    print("Copying and resizing...")

    # 创建不同分辨率图像的目录
    # Resize images.
    os.makedirs(args.source_path + "/images_2", exist_ok=True)  # 50%尺寸的图像目录
    os.makedirs(args.source_path + "/images_4", exist_ok=True)  # 25%尺寸的图像目录
    os.makedirs(args.source_path + "/images_8", exist_ok=True)  # 12.5%尺寸的图像目录
    # 获取原始图像目录中的文件列表
    # Get the list of files in the source directory
    files = os.listdir(args.source_path + "/images")
    # 处理每个图像文件
    # Copy each file from the source directory to the destination directory
    for file in files:
        # 获取源文件的完整路径
        source_file = os.path.join(args.source_path, "images", file)

        # 处理50%尺寸的图像
        destination_file = os.path.join(args.source_path, "images_2", file)
        shutil.copy2(source_file, destination_file)  # 复制源文件到目标目录
        # 使用ImageMagick调整图像尺寸为原来的50%
        exit_code = os.system(magick_command + " mogrify -resize 50% " + destination_file)
        if exit_code != 0:
            logging.error(f"50% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        # 处理25%尺寸的图像
        destination_file = os.path.join(args.source_path, "images_4", file)
        shutil.copy2(source_file, destination_file)  # 复制源文件到目标目录
        # 使用ImageMagick调整图像尺寸为原来的25%
        exit_code = os.system(magick_command + " mogrify -resize 25% " + destination_file)
        if exit_code != 0:
            logging.error(f"25% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

        # 处理12.5%尺寸的图像
        destination_file = os.path.join(args.source_path, "images_8", file)
        shutil.copy2(source_file, destination_file)  # 复制源文件到目标目录
        # 使用ImageMagick调整图像尺寸为原来的12.5%
        exit_code = os.system(magick_command + " mogrify -resize 12.5% " + destination_file)
        if exit_code != 0:
            logging.error(f"12.5% resize failed with code {exit_code}. Exiting.")
            exit(exit_code)

# 处理完成提示
print("Done.")
