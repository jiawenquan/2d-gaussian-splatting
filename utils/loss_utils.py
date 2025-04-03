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
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

# ===== 颜色差异评分表：L1损失 =====
# 就像老师检查你涂色本上每个像素的颜色差多少
# 把所有的差异加起来，算出平均分数
def l1_loss(network_output, gt):
    # network_output：电脑画的图
    # gt：真实的图（ground truth）
    # torch.abs()：计算差异的大小（不管是太深还是太浅）
    # .mean()：计算平均差异
    return torch.abs((network_output - gt)).mean()

# ===== 颜色差异加权评分表：L2损失 =====
# 和L1类似，但会对大错误更加"生气"
# 就像把红色涂成蓝色（大错误）会被扣很多分
# 把深红色涂成浅红色（小错误）只扣一点点分
def l2_loss(network_output, gt):
    # **2：把差异平方，让大错误被惩罚更多
    return ((network_output - gt) ** 2).mean()

# ===== 创建模糊的"印章"：高斯核 =====
# 这个函数创建一个特殊的圆形"印章"
# 中间墨水多，边缘墨水少，用来模糊处理
def gaussian(window_size, sigma):
    # 创建一个一维高斯核（一行数字）
    # 中间的数字最大，两边逐渐变小
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    # 确保所有数字加起来等于1（就像分配100分）
    return gauss / gauss.sum()


# ===== 平滑度评分表：平滑损失 =====
# 确保深度图在应该平滑的地方是平滑的
# 就像告诉小朋友："在同一个物体内部，深度不应该突然变化"
def smooth_loss(disp, img):
    # 检查深度图在水平方向上变化有多大
    # 就像检查从左到右是否有突然的深度跳跃
    grad_disp_x = torch.abs(disp[:,1:-1, :-2] + disp[:,1:-1,2:] - 2 * disp[:,1:-1,1:-1])
    
    # 检查深度图在垂直方向上变化有多大
    # 就像检查从上到下是否有突然的深度跳跃
    grad_disp_y = torch.abs(disp[:,:-2, 1:-1] + disp[:,2:,1:-1] - 2 * disp[:,1:-1,1:-1])
    
    # 检查颜色图在水平方向上变化有多大
    # 颜色变化大的地方通常是物体边缘
    grad_img_x = torch.mean(torch.abs(img[:, 1:-1, :-2] - img[:, 1:-1, 2:]), 0, keepdim=True) * 0.5
    
    # 检查颜色图在垂直方向上变化有多大
    grad_img_y = torch.mean(torch.abs(img[:, :-2, 1:-1] - img[:, 2:, 1:-1]), 0, keepdim=True) * 0.5
    
    # 聪明的部分：在物体边缘（颜色变化大的地方）允许深度变化大
    # 在物体内部（颜色变化小的地方）要求深度变化小
    # 就像说："物体内部应该平滑，物体边缘可以有台阶"
    grad_disp_x *= torch.exp(-grad_img_x)  # 颜色变化大时，这个值接近0，减少惩罚
    grad_disp_y *= torch.exp(-grad_img_y)  # 颜色变化小时，这个值接近1，保持惩罚
    
    # 返回水平和垂直方向平滑度的平均值
    return grad_disp_x.mean() + grad_disp_y.mean()

# ===== 创建模糊的"放大镜"：2D高斯窗口 =====
# 为SSIM创建一个圆形的窗口，用来一次看图像的一小块区域
def create_window(window_size, channel):
    # 先创建一条水平的高斯线（就像一行渐变的墨水）
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    
    # 把这条线变成一个圆形的印章（中间深，边缘浅）
    # mm是矩阵乘法，t()是转置（把行变成列）
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    
    # 为图像的每个颜色通道（红、绿、蓝）复制这个印章
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

# ===== 整体相似性评分：SSIM =====
# 不只看单个像素，而是看小块区域整体感觉
# 就像老师不只看你每个颜色点涂得对不对，而是看整体效果
def ssim(img1, img2, window_size=11, size_average=True):
    # 获取图像有几个颜色通道（通常是3：红、绿、蓝）
    channel = img1.size(-3)
    
    # 创建我们的"放大镜"（高斯窗口）
    # 默认大小是11x11像素的区域
    window = create_window(window_size, channel)

    # 确保"放大镜"在和图像相同的设备上（CPU或GPU）
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    # 用这个"放大镜"计算两张图的相似度
    return _ssim(img1, img2, window, window_size, channel, size_average)

# ===== SSIM的详细计算步骤 =====
# 这是评价两张图片有多相似的复杂公式
# 考虑了亮度、对比度和结构三个方面
def _ssim(img1, img2, window, window_size, channel, size_average=True):
    # 第1步：用"放大镜"计算每个区域的平均亮度
    # 就像看看每个小区域整体有多亮
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    # 计算平均亮度的平方，后面公式会用到
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # 第2步：计算每个区域的对比度（方差）和结构（协方差）
    # 方差：看看区域内亮度变化有多大
    # 协方差：看看两张图的亮度变化是否一致
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    # 加一些小常数，防止除以0（就像避免被0分）
    C1 = 0.01 ** 2  # 亮度比较的稳定值
    C2 = 0.03 ** 2  # 对比度比较的稳定值

    # 第3步：神奇的SSIM公式，综合考虑亮度、对比度和结构
    # 公式解释：
    # - 分子第一项：比较亮度相似性
    # - 分子第二项：比较对比度和结构相似性
    # - 分母：归一化因子
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    # 计算整张图的平均相似度
    # 值越接近1，两张图越相似；越接近0，越不相似
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

