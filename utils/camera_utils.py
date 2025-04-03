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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    """
    加载相机参数并处理图像分辨率
    
    参数:
        args: 包含程序参数的对象
        id: 相机ID
        cam_info: 相机信息对象
        resolution_scale: 分辨率缩放因子
    """
    orig_w, orig_h = cam_info.image.size  # 获取原始图像尺寸

    # 根据参数设置处理图像分辨率
    if args.resolution in [1, 2, 4, 8]:
        # 整数分辨率设置，直接将原始分辨率除以缩放因子
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            # 对于非常大的图像，自动缩放到1600像素宽
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            # 使用指定的分辨率缩放因子
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    # 处理图像的通道数，如果大于3通道，分离RGB和Alpha通道
    if len(cam_info.image.split()) > 3:
        import torch
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in cam_info.image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(cam_info.image.split()[3], resolution)  # 获取Alpha通道作为掩码
        gt_image = resized_image_rgb
    else:
        # 处理标准的3通道RGB图像
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb

    # 创建并返回Camera对象
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    """
    从相机信息列表创建相机对象列表
    
    参数:
        cam_infos: 相机信息列表
        resolution_scale: 分辨率缩放因子
        args: 程序参数
    """
    camera_list = []

    # 为每个相机信息创建相机对象
    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    """
    将相机参数转换为JSON格式
    
    参数:
        id: 相机ID
        camera: 相机对象
    """
    # 构建从世界坐标到相机坐标的变换矩阵
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    # 计算相机的世界坐标位置和旋转
    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]  # 相机在世界坐标中的位置
    rot = W2C[:3, :3]  # 相机在世界坐标中的旋转
    
    # 将旋转矩阵转换为Python列表格式
    serializable_array_2d = [x.tolist() for x in rot]
    
    # 创建JSON格式的相机条目
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),  # 垂直方向的焦距
        'fx' : fov2focal(camera.FovX, camera.width)    # 水平方向的焦距
    }
    return camera_entry