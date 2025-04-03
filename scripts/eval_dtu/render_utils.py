import numpy as np
import imageio
import skimage
import cv2
import torch
from torch.nn import functional as F


def get_psnr(img1, img2, normalize_rgb=False):
    """
    计算两张图像之间的峰值信噪比(PSNR)
    
    参数:
        img1: 第一张图像
        img2: 第二张图像
        normalize_rgb: 是否将图像从[-1,1]范围归一化到[0,1]范围
        
    返回:
        PSNR值，越高表示图像质量越好
    """
    if normalize_rgb: # [-1,1] --> [0,1]
        img1 = (img1 + 1.) / 2.
        img2 = (img2 + 1. ) / 2.

    mse = torch.mean((img1 - img2) ** 2)  # 计算均方误差
    psnr = -10. * torch.log(mse) / torch.log(torch.Tensor([10.]).cuda())  # 计算PSNR

    return psnr


def load_rgb(path, normalize_rgb = False):
    """
    加载RGB图像
    
    参数:
        path: 图像路径
        normalize_rgb: 是否将图像从[0,1]范围归一化到[-1,1]范围
        
    返回:
        加载并处理后的图像，形状为[3,H,W]
    """
    img = imageio.imread(path)  # 读取图像
    img = skimage.img_as_float32(img)  # 转换为float32类型

    if normalize_rgb: # [-1,1] --> [0,1]
        img -= 0.5  # 减去0.5
        img *= 2.  # 乘以2，实现归一化到[-1,1]
    img = img.transpose(2, 0, 1)  # 将HWC转换为CHW格式
    return img


def load_K_Rt_from_P(filename, P=None):
    """
    从投影矩阵P分解出内参矩阵K和外参矩阵[R|t]
    
    参数:
        filename: 可选，包含投影矩阵的文件路径
        P: 可选，直接提供的投影矩阵，如果为None则从文件加载
        
    返回:
        intrinsics: 相机内参矩阵(4x4)
        pose: 相机位姿矩阵(4x4)，从相机空间到世界空间的变换
    """
    if P is None:
        # 从文件加载投影矩阵
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    # 使用OpenCV分解投影矩阵
    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]  # 内参矩阵
    R = out[1]  # 旋转矩阵
    t = out[2]  # 平移向量

    # 归一化内参矩阵
    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    # 构建相机位姿矩阵(从相机到世界坐标)
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()  # 旋转矩阵的转置
    pose[:3,3] = (t[:3] / t[3])[:,0]  # 平移向量

    return intrinsics, pose


def get_camera_params(uv, pose, intrinsics):
    """
    根据像素坐标，相机位姿和内参，计算射线方向和相机位置
    
    参数:
        uv: 像素坐标，形状为[batch_size, num_samples, 2]
        pose: 相机位姿，可以是四元数表示或矩阵表示
        intrinsics: 相机内参
        
    返回:
        ray_dirs: 射线方向，形状为[batch_size, num_samples, 3]
        cam_loc: 相机位置，形状为[batch_size, 3]
    """
    if pose.shape[1] == 7: #In case of quaternion vector representation
        # 如果位姿是四元数表示（前4个元素是四元数，后3个是位置）
        cam_loc = pose[:, 4:]  # 相机位置
        R = quat_to_rot(pose[:,:4])  # 从四元数转换为旋转矩阵
        p = torch.eye(4).repeat(pose.shape[0],1,1).cuda().float()  # 创建单位矩阵
        p[:, :3, :3] = R  # 设置旋转部分
        p[:, :3, 3] = cam_loc  # 设置平移部分
    else: # In case of pose matrix representation
        # 如果位姿是矩阵表示
        cam_loc = pose[:, :3, 3]  # 相机位置（矩阵的第4列前3行）
        p = pose

    batch_size, num_samples, _ = uv.shape

    # 创建深度值为1的平面
    depth = torch.ones((batch_size, num_samples)).cuda()
    x_cam = uv[:, :, 0].view(batch_size, -1)  # x坐标
    y_cam = uv[:, :, 1].view(batch_size, -1)  # y坐标
    z_cam = depth.view(batch_size, -1)  # z坐标（深度）

    # 提升到3D空间
    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)

    # 转置以进行批量矩阵乘法
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

    # 将相机坐标转换为世界坐标
    world_coords = torch.bmm(p, pixel_points_cam).permute(0, 2, 1)[:, :, :3]
    # 计算射线方向（从相机位置到世界坐标的向量）
    ray_dirs = world_coords - cam_loc[:, None, :]
    # 归一化射线方向
    ray_dirs = F.normalize(ray_dirs, dim=2)

    return ray_dirs, cam_loc


def get_camera_for_plot(pose):
    """
    从相机位姿中提取相机位置和方向，用于绘图
    
    参数:
        pose: 相机位姿，可以是四元数表示或矩阵表示
        
    返回:
        cam_loc: 相机位置
        cam_dir: 相机方向（相机的z轴）
    """
    if pose.shape[1] == 7: #In case of quaternion vector representation
        # 如果位姿是四元数表示
        cam_loc = pose[:, 4:].detach()  # 相机位置
        R = quat_to_rot(pose[:,:4].detach())  # 从四元数转换为旋转矩阵
    else: # In case of pose matrix representation
        # 如果位姿是矩阵表示
        cam_loc = pose[:, :3, 3]  # 相机位置
        R = pose[:, :3, :3]  # 旋转矩阵
    cam_dir = R[:, :3, 2]  # 相机的z轴（朝向）
    return cam_loc, cam_dir


def lift(x, y, z, intrinsics):
    """
    将像素坐标提升到3D空间（反投影）
    
    参数:
        x: 归一化的x坐标
        y: 归一化的y坐标
        z: 深度值
        intrinsics: 相机内参矩阵
        
    返回:
        包含3D点的齐次坐标的张量，形状为[batch_size, num_points, 4]
    """
    # 解析内参
    intrinsics = intrinsics.cuda()
    fx = intrinsics[:, 0, 0]  # 焦距x
    fy = intrinsics[:, 1, 1]  # 焦距y
    cx = intrinsics[:, 0, 2]  # 主点x
    cy = intrinsics[:, 1, 2]  # 主点y
    sk = intrinsics[:, 0, 1]  # 倾斜因子

    # 计算3D点的x坐标（考虑倾斜因子）
    x_lift = (x - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z
    # 计算3D点的y坐标
    y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

    # 创建齐次坐标 [x,y,z,1]
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z).cuda()), dim=-1)


def quat_to_rot(q):
    """
    将四元数转换为旋转矩阵
    
    参数:
        q: 四元数，形状为[batch_size, 4]
        
    返回:
        旋转矩阵，形状为[batch_size, 3, 3]
    """
    batch_size, _ = q.shape
    # 归一化四元数
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3,3)).cuda()
    # 提取四元数的各个分量
    qr=q[:,0]  # 实部
    qi = q[:, 1]  # i分量
    qj = q[:, 2]  # j分量
    qk = q[:, 3]  # k分量
    
    # 使用四元数到旋转矩阵的转换公式
    R[:, 0, 0]=1-2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj *qi -qk*qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1-2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2*(qj*qk - qi*qr)
    R[:, 2, 0] = 2 * (qk * qi-qj * qr)
    R[:, 2, 1] = 2 * (qj*qk + qi*qr)
    R[:, 2, 2] = 1-2 * (qi**2 + qj**2)
    return R


def rot_to_quat(R):
    """
    将旋转矩阵转换为四元数
    
    参数:
        R: 旋转矩阵，形状为[batch_size, 3, 3]
        
    返回:
        四元数，形状为[batch_size, 4]
    """
    batch_size, _,_ = R.shape
    q = torch.ones((batch_size, 4)).cuda()

    # 提取旋转矩阵的各个元素
    R00 = R[:, 0,0]
    R01 = R[:, 0, 1]
    R02 = R[:, 0, 2]
    R10 = R[:, 1, 0]
    R11 = R[:, 1, 1]
    R12 = R[:, 1, 2]
    R20 = R[:, 2, 0]
    R21 = R[:, 2, 1]
    R22 = R[:, 2, 2]

    # 使用旋转矩阵到四元数的转换公式
    q[:,0]=torch.sqrt(1.0+R00+R11+R22)/2  # 实部
    q[:, 1]=(R21-R12)/(4*q[:,0])  # i分量
    q[:, 2] = (R02 - R20) / (4 * q[:, 0])  # j分量
    q[:, 3] = (R10 - R01) / (4 * q[:, 0])  # k分量
    return q


def get_sphere_intersections(cam_loc, ray_directions, r = 1.0):
    """
    计算射线与球体的交点距离
    
    参数:
        cam_loc: 相机位置，形状为[n_rays, 3]
        ray_directions: 射线方向，形状为[n_rays, 3]
        r: 球体半径，默认为1.0
        
    返回:
        sphere_intersections: 近点和远点的交点距离，形状为[n_rays, 2]
    """
    # 计算射线方向与相机位置的点积
    ray_cam_dot = torch.bmm(ray_directions.view(-1, 1, 3),
                            cam_loc.view(-1, 3, 1)).squeeze(-1)
    # 计算判别式
    under_sqrt = ray_cam_dot ** 2 - (cam_loc.norm(2, 1, keepdim=True) ** 2 - r ** 2)

    # 判断是否所有射线都与球体相交
    if (under_sqrt <= 0).sum() > 0:
        print('BOUNDING SPHERE PROBLEM!')
        exit()

    # 计算交点距离（近点和远点）
    sphere_intersections = torch.sqrt(under_sqrt) * torch.Tensor([-1, 1]).cuda().float() - ray_cam_dot
    # 确保所有交点距离都大于等于0
    sphere_intersections = sphere_intersections.clamp_min(0.0)

    return sphere_intersections