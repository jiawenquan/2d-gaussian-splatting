import os
from argparse import ArgumentParser

# 定义TNT(Tanks and Temples)数据集中的360度场景列表
tnt_360_scenes = ['Barn', 'Caterpillar', 'Ignatius', 'Truck']
# 定义TNT数据集中的大型场景列表
tnt_large_scenes = ['Meetingroom', 'Courthouse']

# 设置命令行参数解析器
parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")  # 跳过训练阶段的标志
parser.add_argument("--skip_rendering", action="store_true")  # 跳过渲染阶段的标志
parser.add_argument("--skip_metrics", action="store_true")  # 跳过指标计算阶段的标志
parser.add_argument("--output_path", default="./eval/tnt")  # 输出路径，默认为./eval/tnt
parser.add_argument('--TNT_data', "-TNT_data", required=True, type=str)  # TNT数据集路径，必需参数
args, _ = parser.parse_known_args()  # 解析已知参数，忽略未知参数

# 如果不跳过指标计算，则需要GT(Ground Truth)数据路径
if not args.skip_metrics:
    parser.add_argument('--TNT_GT', required=True, type=str)  # TNT真值数据路径
    args = parser.parse_args()  # 重新解析所有参数


# 训练阶段
if not args.skip_training:
    # 定义所有场景共用的训练参数
    common_args = " --quiet --test_iterations -1 --depth_ratio 1.0 -r 2 "
    
    # 处理360度场景
    for scene in tnt_360_scenes:
        source = args.TNT_data + "/" + scene  # 数据源路径
        print("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args + ' --lambda_dist 100')
        # 执行训练命令
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)

    # 处理大型场景
    for scene in tnt_large_scenes:
        source = args.TNT_data + "/" + scene  # 数据源路径
        print("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args+ ' --lambda_dist 10')
        # 执行训练命令
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)


# 渲染阶段
if not args.skip_rendering:
    all_sources = []  # 用于存储所有数据源路径的列表
    common_args = " --quiet --depth_ratio 1.0 "  # 定义所有场景共用的渲染参数

    # 处理360度场景的渲染
    for scene in tnt_360_scenes:
        source = args.TNT_data + "/" + scene  # 数据源路径
        print("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args + ' --num_cluster 1 --voxel_size 0.004 --sdf_trunc 0.016 --depth_trunc 3.0')
        # 执行渲染命令，带有特定的体素大小和截断参数
        os.system("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args + '  --num_cluster 1 --voxel_size 0.004 --sdf_trunc 0.016 --depth_trunc 3.0')

    # 处理大型场景的渲染
    for scene in tnt_large_scenes:
        source = args.TNT_data + "/" + scene  # 数据源路径
        print("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args + ' --num_cluster 1 --voxel_size 0.006 --sdf_trunc 0.024 --depth_trunc 4.5')
        # 执行渲染命令，为大型场景使用不同的参数设置
        os.system("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args + ' --num_cluster 1 --voxel_size 0.006 --sdf_trunc 0.024 --depth_trunc 4.5')

# 评估指标计算阶段
if not args.skip_metrics:
    script_dir = os.path.dirname(os.path.abspath(__file__))  # 获取当前脚本所在目录
    all_scenes = tnt_360_scenes + tnt_large_scenes  # 合并所有场景列表

    # 对每个场景计算评估指标
    for scene in all_scenes:
        ply_file = f"{args.output_path}/{scene}/train/ours_{iteration}/fuse_post.ply"  # PLY文件路径（存在变量未定义的问题）
        string = f"OMP_NUM_THREADS=4 python {script_dir}/eval_tnt/run.py " + \
            f"--dataset-dir {args.TNT_GT}/{scene} " + \
            f"--traj-path {args.TNT_data}/{scene}/{scene}_COLMAP_SfM.log " + \
            f"--ply-path {ply_file}"
        print(string)
        # 执行评估命令
        os.system(string)