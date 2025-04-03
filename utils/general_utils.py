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

import torch
import sys
from datetime import datetime
import numpy as np
import random

# Sigmoid的逆函数，用于将(0,1)范围的值转换为(-∞,+∞)范围
def inverse_sigmoid(x):
    return torch.log(x/(1-x))

# 将PIL图像转换为torch张量，调整图像大小并进行归一化处理
def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)  # 将HWC转换为CHW格式
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)  # 将灰度图像转换为单通道

# 指数学习率衰减函数，用于优化过程中学习率的调整
def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

# 提取下三角矩阵中的元素，用于处理协方差矩阵或不确定性矩阵
def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

# 提取对称矩阵中的元素，实际是调用strip_lowerdiag
def strip_symmetric(sym):
    return strip_lowerdiag(sym)

# 根据四元数构建旋转矩阵
def build_rotation(r):
    # 归一化四元数
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
    q = r / norm[:, None]

    # 初始化旋转矩阵
    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    # 提取四元数的各个分量
    r = q[:, 0]  # 实部
    x = q[:, 1]  # i分量
    y = q[:, 2]  # j分量
    z = q[:, 3]  # k分量

    # 根据四元数公式构建旋转矩阵
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

# 构建缩放旋转矩阵，结合了缩放和旋转
def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)  # 构建旋转矩阵

    # 设置缩放因子
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    # 将旋转应用于缩放矩阵
    L = R @ L
    return L

# 为程序执行创建安全状态，包括设置输出重定向和随机种子
def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    # 重定向标准输出
    sys.stdout = F(silent)

    # 设置随机种子以确保结果可重现
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))




# 从方向向量批量创建旋转矩阵
def create_rotation_matrix_from_direction_vector_batch(direction_vectors):
    # 归一化方向向量
    direction_vectors = direction_vectors / torch.norm(direction_vectors, dim=-1, keepdim=True)
    # 创建与方向向量不共线的任意向量
    v1 = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32).to(direction_vectors.device).expand(direction_vectors.shape[0], -1).clone()
    is_collinear = torch.all(torch.abs(direction_vectors - v1) < 1e-5, dim=-1)
    v1[is_collinear] = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32).to(direction_vectors.device)

    # 计算第一个正交向量
    v1 = torch.cross(direction_vectors, v1)
    v1 = v1 / (torch.norm(v1, dim=-1, keepdim=True))
    # 计算第二个正交向量（通过叉积）
    v2 = torch.cross(direction_vectors, v1)
    v2 = v2 / (torch.norm(v2, dim=-1, keepdim=True))
    # 创建旋转矩阵，以方向向量作为最后一列
    rotation_matrices = torch.stack((v1, v2, direction_vectors), dim=-1)
    return rotation_matrices

# 将法线转换为旋转的函数注释（未实现）
# from kornia.geometry import conversions
# def normal_to_rotation(normals):
#     rotations = create_rotation_matrix_from_direction_vector_batch(normals)
#     rotations = conversions.rotation_matrix_to_quaternion(rotations,eps=1e-5, order=conversions.QuaternionCoeffOrder.WXYZ)
#     return rotations


# 生成带有颜色映射的图像
def colormap(img, cmap='jet'):
    import matplotlib.pyplot as plt
    W, H = img.shape[:2]
    dpi = 300
    fig, ax = plt.subplots(1, figsize=(H/dpi, W/dpi), dpi=dpi)
    im = ax.imshow(img, cmap=cmap)
    ax.set_axis_off()
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = torch.from_numpy(data / 255.).float().permute(2,0,1)
    plt.close()
    return img