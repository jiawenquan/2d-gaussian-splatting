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
import matplotlib.pyplot as plt
import torch.nn.functional as F

# ===== 计算画得有多差：均方误差(MSE) =====
# 就像比较两张图，看每个像素差多少，然后算平均分
# 差异越大，分数越高（这是个"错误分数"）
def mse(img1, img2):
    # img1和img2：两张要比较的图片
    # ** 2：计算差异的平方（放大差异）
    # view：把图片展平成一行数字
    # mean：计算平均值
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

# ===== 图像质量分数：峰值信噪比(PSNR) =====
# 这是一种常用的图像质量评分，分数越高越好
# 就像考试分数：100分是完美相同，0分是完全不同
def psnr(img1, img2):
    # 先计算均方误差（和上面的函数一样）
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    # 用公式把误差转换成分数：20*log10(1.0/√误差)
    # 图像完全相同时，误差=0，PSNR=无穷大（非常高的分数）
    # 图像差异越大，分数越低
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

# ===== 找出图像的边缘：梯度图 =====
# 就像用铅笔描出一幅画中所有物体的轮廓
# 帮助电脑识别图像中哪里是物体的边缘
def gradient_map(image):
    # Sobel算子：边缘检测的"魔法眼镜"
    # 这是两个特殊的"印章"，帮助找出水平和垂直的边缘
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()/4
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().unsqueeze(0).unsqueeze(0).cuda()/4
    
    # 用"魔法眼镜"检查图像的每个部分
    # 寻找水平方向的边缘（左右颜色变化）
    grad_x = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_x, padding=1) for i in range(image.shape[0])])
    # 寻找垂直方向的边缘（上下颜色变化）
    grad_y = torch.cat([F.conv2d(image[i].unsqueeze(0), sobel_y, padding=1) for i in range(image.shape[0])])
    
    # 计算总边缘强度（结合水平和垂直方向）
    # 就像画轮廓时，笔画有粗有细，取决于边缘的明显程度
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    magnitude = magnitude.norm(dim=0, keepdim=True)

    return magnitude

# ===== 给单色图添加好看的颜色：颜色映射 =====
# 就像给黑白照片上色，让它看起来更漂亮
# 比如把深度图从灰色变成彩虹色，方便我们看清深浅变化
def colormap(map, cmap="turbo"):
    # 获取一个漂亮的颜色表（默认是"turbo"彩虹色）
    # 就像准备一盒彩色铅笔，从蓝色到红色
    colors = torch.tensor(plt.cm.get_cmap(cmap).colors).to(map.device)
    
    # 将数据调整到0到1之间（归一化）
    # 就像把各种身高的人按比例变成0到100厘米
    map = (map - map.min()) / (map.max() - map.min())
    
    # 将0-1的值变成0-255的颜色索引
    # 就像选择彩色铅笔盒中的第几号铅笔
    map = (map * 255).round().long().squeeze()
    
    # 用索引从颜色表中取出对应的颜色
    # 并调整颜色通道的顺序（红绿蓝）
    map = colors[map].permute(2,0,1)
    return map

# ===== 选择不同的查看方式：渲染不同类型的图像 =====
# 就像戴上不同的"魔法眼镜"，看到物体的不同方面
# 可以看颜色、深度、边缘、表面方向等
def render_net_image(render_pkg, render_items, render_mode, camera):
    # 根据用户选择的渲染模式决定输出什么类型的图像
    output = render_items[render_mode].lower()
    
    if output == 'alpha':
        # 透明度眼镜：看物体的透明和不透明部分
        # 白色表示不透明，黑色表示透明
        net_image = render_pkg["rend_alpha"]
        
    elif output == 'normal':
        # 表面方向眼镜：看每个点表面朝向哪个方向
        # 不同方向用不同颜色表示（红、绿、蓝）
        net_image = render_pkg["rend_normal"]
        # 将-1到1的值调整到0到1（让颜色更好看）
        net_image = (net_image+1)/2
        
    elif output == 'depth':
        # 深度眼镜：看物体的远近
        # 近的部分亮，远的部分暗
        net_image = render_pkg["surf_depth"]
        
    elif output == 'edge':
        # 边缘眼镜：只看物体的轮廓线
        # 就像素描画，只有线条没有颜色填充
        net_image = gradient_map(render_pkg["render"])
        
    elif output == 'curvature':
        # 曲率眼镜：看表面的弯曲程度
        # 平坦的地方暗，弯曲的地方亮
        net_image = render_pkg["rend_normal"]
        net_image = (net_image+1)/2
        net_image = gradient_map(net_image)
        
    else:
        # 普通眼镜：看物体的真实颜色（默认模式）
        net_image = render_pkg["render"]

    # 如果图像只有一个通道（黑白图），给它添加漂亮的颜色
    # 让单色的深度图或边缘图变得更容易辨认
    if net_image.shape[0]==1:
        net_image = colormap(net_image)
        
    return net_image