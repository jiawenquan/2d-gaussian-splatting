import os
from argparse import ArgumentParser

# 定义DTU (Danish Technological University)数据集中的场景列表
dtu_scenes = ['scan24', 'scan37', 'scan40', 'scan55', 'scan63', 'scan65', 'scan69', 'scan83', 'scan97', 'scan105', 'scan106', 'scan110', 'scan114', 'scan118', 'scan122']

# 设置命令行参数解析器
parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")  # 跳过训练阶段的标志
parser.add_argument("--skip_rendering", action="store_true")  # 跳过渲染阶段的标志
parser.add_argument("--skip_metrics", action="store_true")  # 跳过指标计算阶段的标志
parser.add_argument("--output_path", default="./eval/dtu")  # 输出路径，默认为./eval/dtu
parser.add_argument('--dtu', "-dtu", required=True, type=str)  # DTU数据集路径，必需参数
args, _ = parser.parse_known_args()  # 解析已知参数，忽略未知参数

# 初始化场景列表
all_scenes = []
all_scenes.extend(dtu_scenes)  # 将DTU场景添加到列表中

# 如果不跳过指标计算，则需要DTU官方数据路径
if not args.skip_metrics:
    parser.add_argument('--DTU_Official', "-DTU", required=True, type=str)  # DTU官方数据集路径
    args = parser.parse_args()  # 重新解析所有参数


# 训练阶段
if not args.skip_training:
    # 定义所有场景共用的训练参数
    common_args = " --quiet --test_iterations -1 --depth_ratio 1.0 -r 2 --lambda_dist 1000"
    for scene in dtu_scenes:
        source = args.dtu + "/" + scene  # 数据源路径
        print("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        # 执行训练命令
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)


# 渲染阶段
if not args.skip_rendering:
    all_sources = []  # 用于存储所有数据源路径的列表
    # 定义所有场景共用的渲染参数
    common_args = " --quiet --skip_train --depth_ratio 1.0 --num_cluster 1 --voxel_size 0.004 --sdf_trunc 0.016 --depth_trunc 3.0"
    for scene in dtu_scenes:
        source = args.dtu + "/" + scene  # 数据源路径
        print("python render.py --iteration 30000 -s " + source + " -m" + args.output_path + "/" + scene + common_args)
        # 执行渲染命令
        os.system("python render.py --iteration 30000 -s " + source + " -m" + args.output_path + "/" + scene + common_args)


# 评估指标计算阶段
if not args.skip_metrics:
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在目录
    for scene in dtu_scenes:
        scan_id = scene[4:]  # 提取场景ID(去掉'scan'前缀)
        ply_file = f"{args.output_path}/{scene}/train/ours_30000/"  # PLY文件目录路径
        iteration = 30000  # 迭代次数(注意：此变量未在后续代码中使用)
        # 构建评估命令
        string = f"python {script_dir}/eval_dtu/evaluate_single_scene.py " + \
            f"--input_mesh {args.output_path}/{scene}/train/ours_30000/fuse_post.ply " + \
            f"--scan_id {scan_id} --output_dir {script_dir}/tmp/scan{scan_id} " + \
            f"--mask_dir {args.dtu} " + \
            f"--DTU {args.DTU_Official}"
        
        # 执行评估命令
        os.system(string)