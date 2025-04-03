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

# ===== 导入我们需要的魔法工具 =====
import torch  # PyTorch：电脑思考的大脑
import math   # 数学工具箱
from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer  # 特殊的绘画工具
from scene.gaussian_model import GaussianModel  # 我们的小星星模型
from utils.sh_utils import eval_sh  # 颜色转换工具
from utils.point_utils import depth_to_normal  # 深度转法线工具

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    # ===== 把小星星变成美丽的图片 =====
    这个函数就像一台神奇的照相机，它能把我们的小星星拍成漂亮的照片！
    
    想象一下：
    - 你有一盒闪亮的小星星（我们的高斯模型）
    - 一台特殊的相机（viewpoint_camera）
    - 和一张背景布（bg_color）
    
    这个函数就是按下相机的快门按钮，拍下小星星的样子！
    
    参数:
        viewpoint_camera: 相机（决定从哪个角度看小星星）
        pc: 小星星盒子（我们的高斯模型）
        pipe: 渲染管道（一系列渲染设置）
        bg_color: 背景颜色（必须在GPU上）
        scaling_modifier: 大小调整系数（让小星星变大或变小）
        override_color: 覆盖颜色（如果想给所有小星星统一上色）
    """
 
    # 创建一个空的屏幕坐标张量
    # 就像准备一张透明纸，等着记录每个小星星在照片上的位置
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # 设置相机的视野角度
    # 就像调整相机的镜头，决定能看到多宽的范围
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)  # 水平视野
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)  # 垂直视野

    # 创建光栅化设置（就像设置相机的各种参数）
    # 光栅化就是把3D小星星变成2D图像的过程，就像照相
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),  # 照片高度
        image_width=int(viewpoint_camera.image_width),    # 照片宽度
        tanfovx=tanfovx,  # 水平视野范围
        tanfovy=tanfovy,  # 垂直视野范围
        bg=bg_color,      # 背景颜色
        scale_modifier=scaling_modifier,  # 小星星大小调整
        viewmatrix=viewpoint_camera.world_view_transform,  # 相机位置矩阵
        projmatrix=viewpoint_camera.full_proj_transform,   # 投影矩阵
        sh_degree=pc.active_sh_degree,  # 颜色复杂度
        campos=viewpoint_camera.camera_center,  # 相机中心
        prefiltered=False,  # 不预过滤
        debug=False,        # 不调试
        # pipe.debug
    )

    # 创建光栅化器（就像准备好相机）
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # 获取小星星的3D位置和2D屏幕位置
    means3D = pc.get_xyz   # 3D空间中的位置
    means2D = screenspace_points  # 屏幕上的位置
    opacity = pc.get_opacity  # 不透明度（星星有多亮）

    # 处理小星星的形状信息
    # 这就像告诉相机每个小星星有多大、朝向哪里
    scales = None      # 大小
    rotations = None   # 旋转
    cov3D_precomp = None  # 预计算的形状
    
    # 如果需要在Python中计算3D形状，就计算它
    # 就像提前计算每个小星星在照片中的形状
    if pipe.compute_cov3D_python:
        # 当前不支持使用预计算协方差的法线一致性损失
        splat2world = pc.get_covariance(scaling_modifier)  # 获取协方差
        
        # 获取相机尺寸和深度范围
        W, H = viewpoint_camera.image_width, viewpoint_camera.image_height
        near, far = viewpoint_camera.znear, viewpoint_camera.zfar
        
        # 创建NDC到像素的变换矩阵
        # 就像创建一个地图，告诉我们如何从3D空间转到2D照片
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, (W-1) / 2],
            [0, H / 2, 0, (H-1) / 2],
            [0, 0, far-near, near],
            [0, 0, 0, 1]]).float().cuda().T
        
        # 计算世界到像素的变换
        world2pix = viewpoint_camera.full_proj_transform @ ndc2pix
        
        # 计算预计算的3D协方差（形状信息）
        cov3D_precomp = (splat2world[:, [0,1,3]] @ world2pix[:,[0,1,3]]).permute(0,2,1).reshape(-1, 9)  # 列主序
    else:
        # 否则，就使用原始的缩放和旋转参数
        scales = pc.get_scaling     # 小星星的大小
        rotations = pc.get_rotation  # 小星星的方向
    
    # 处理小星星的颜色信息
    # 这就像决定每个小星星在照片中应该是什么颜色
    pipe.convert_SHs_python = False
    shs = None  # 球谐系数
    colors_precomp = None  # 预计算的颜色
    
    # 如果没有提供覆盖颜色
    if override_color is None:
        # 如果需要在Python中计算颜色
        if pipe.convert_SHs_python:
            # 从视角方向计算RGB颜色
            # 就像根据我们看小星星的角度计算它应该是什么颜色
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)  # 确保颜色是正数
        else:
            # 否则，直接使用特征作为球谐系数
            shs = pc.get_features
    else:
        # 如果提供了覆盖颜色，就使用它
        colors_precomp = override_color
    
    # 使用光栅化器"拍照"！
    # 这是魔法发生的地方，3D小星星变成2D图像
    rendered_image, radii, allmap = rasterizer(
        means3D = means3D,         # 3D位置
        means2D = means2D,         # 2D位置
        shs = shs,                 # 颜色球谐系数
        colors_precomp = colors_precomp,  # 预计算的颜色
        opacities = opacity,       # 不透明度
        scales = scales,           # 大小
        rotations = rotations,     # 方向
        cov3D_precomp = cov3D_precomp  # 预计算的形状
    )
    
    # 收集渲染结果
    # 就像整理我们拍好的照片及相关信息
    rets = {
        "render": rendered_image,            # 渲染的图像（最终照片）
        "viewspace_points": means2D,         # 视空间点（小星星在照片上的位置）
        "visibility_filter": radii > 0,      # 可见性过滤器（哪些小星星实际出现在照片中）
        "radii": radii,                      # 半径（小星星在照片上有多大）
    }

    # === 收集额外信息，用于提高照片质量 ===

    # 获取透明度图（Alpha通道）
    # 就像知道照片上每个位置有多少"星星亮光"
    render_alpha = allmap[1:2]

    # 获取法线图（表面方向）
    # 就像知道每个小星星的表面朝向哪里
    render_normal = allmap[2:5]
    # 将法线从视图空间转换到世界空间
    render_normal = (render_normal.permute(1,2,0) @ (viewpoint_camera.world_view_transform[:3,:3].T)).permute(2,0,1)
    
    # 获取中值深度图（小星星到相机的距离）
    # 就像知道每个小星星离相机有多远（中间值）
    render_depth_median = allmap[5:6]
    render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)  # 处理无效值

    # 获取期望深度图
    # 就像知道每个小星星离相机有多远（加权平均）
    render_depth_expected = allmap[0:1]
    render_depth_expected = (render_depth_expected / render_alpha)
    render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)  # 处理无效值
    
    # 获取深度失真图
    # 就像知道深度估计有多不确定
    render_dist = allmap[6:7]

    # 计算表面深度
    # 根据场景类型选择合适的深度计算方式
    # 就像选择最合适的方式来表示小星星的距离
    surf_depth = render_depth_expected * (1-pipe.depth_ratio) + (pipe.depth_ratio) * render_depth_median
    
    # 从深度估计表面法线
    # 就像根据物体的深度形状，猜测它的表面朝向
    surf_normal = depth_to_normal(viewpoint_camera, surf_depth)
    surf_normal = surf_normal.permute(2,0,1)
    # 应用透明度
    surf_normal = surf_normal * (render_alpha).detach()

    # 将所有额外信息添加到结果中
    # 就像在照片背后附上所有技术细节
    rets.update({
            'rend_alpha': render_alpha,       # 透明度图
            'rend_normal': render_normal,     # 法线图
            'rend_dist': render_dist,         # 深度失真图
            'surf_depth': surf_depth,         # 表面深度图
            'surf_normal': surf_normal,       # 表面法线图
    })

    # 返回渲染结果（照片及所有相关信息）
    return rets