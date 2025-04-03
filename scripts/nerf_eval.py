# training scripts for the nerf-synthetic datasets
# this script is adopted from GOF
# https://github.com/autonomousvision/gaussian-opacity-fields/blob/main/scripts/run_nerf_synthetic.py
import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import time
import itertools

# 定义NeRF合成数据集中的场景列表
scenes = ["ship", "drums", "ficus", "hotdog", "lego", "materials", "mic", "chair"]

# 定义缩放因子列表
factors = [1]

# 输出目录路径
output_dir = "output/exp_nerf_synthetic"

# 数据集目录路径
dataset_dir = "data/nerf_synthetic"

# 是否为干运行(不实际执行命令)
dry_run = False

# 排除的GPU设备集合
excluded_gpus = set([])


# 创建场景和因子的所有组合作为待处理任务
jobs = list(itertools.product(scenes, factors))

def train_scene(gpu, scene, factor):
    # 构建训练命令，包括设置线程数、CUDA设备、数据源路径等参数
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py -s {dataset_dir}/{scene} -m {output_dir}/{scene} --eval --white_background --lambda_normal 0.0 --port {6209+int(gpu)}"
    print(cmd)
    if not dry_run:
        os.system(cmd)  # 执行训练命令

    # 构建渲染命令，跳过训练和网格生成
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {output_dir}/{scene} --skip_train --skip_mesh"
    print(cmd)
    if not dry_run:
        os.system(cmd)  # 执行渲染命令
        
    # 构建指标计算命令
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python metrics.py -m {output_dir}/{scene}"
    print(cmd)
    if not dry_run:
        os.system(cmd)  # 执行指标计算命令
    
    return True

    
def worker(gpu, scene, factor):
    # 在指定的GPU上启动场景训练任务
    print(f"Starting job on GPU {gpu} with scene {scene}\n")
    train_scene(gpu, scene, factor)
    print(f"Finished job on GPU {gpu} with scene {scene}\n")
    # 此工作函数启动一个作业并在完成时返回
    
    
def dispatch_jobs(jobs, executor):
    # 存储未来任务到作业的映射
    future_to_job = {}
    # 存储已预留但可能尚未激活的GPU
    reserved_gpus = set()  # GPUs that are slated for work but may not be active yet

    # 只要还有待处理作业或正在运行的作业，就继续循环
    while jobs or future_to_job:
        # 获取所有可用GPU列表，不包括已预留的和已排除的
        all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=0.5, maxLoad=0.5))
        available_gpus = list(all_available_gpus - reserved_gpus - excluded_gpus)

        # 在可用GPU上启动新任务
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)  # 取出一个可用GPU
            job = jobs.pop(0)  # 取出一个待处理任务
            future = executor.submit(worker, gpu, *job)  # 提交工作函数到执行器，将job解包为参数
            future_to_job[future] = (gpu, job)  # 记录future与job的对应关系

            reserved_gpus.add(gpu)  # 将此GPU标记为已预留，直到任务开始处理

        # 检查已完成的任务，并从运行中的任务列表中移除它们
        # 同时释放它们使用的GPU
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)  # 移除与已完成future关联的job
            gpu = job[0]  # GPU是job元组的第一个元素
            reserved_gpus.discard(gpu)  # 释放此GPU
            print(f"Job {job} has finished., rellasing GPU {gpu}")
        # (可选)可以在这里引入小延迟，防止当没有可用GPU时，循环过快旋转
        time.sleep(5)
        
    print("All jobs have been processed.")


# 使用ThreadPoolExecutor管理线程池
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor)