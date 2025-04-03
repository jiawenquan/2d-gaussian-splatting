#!/usr/bin/env python
# coding=utf-8
import torch

def rotation_matrix(a, b):
    """Compute the rotation matrix that rotates vector a to vector b.

    Args:
        a: The vector to rotate.
        b: The vector to rotate to.
    Returns:
        The rotation matrix.
    """
    # 计算从向量a到向量b的旋转矩阵
    
    # 将输入向量归一化
    a = a / torch.linalg.norm(a)
    b = b / torch.linalg.norm(b)
    
    # 计算叉积，得到旋转轴
    v = torch.cross(a, b)
    
    # 计算点积，得到两向量夹角的余弦值
    c = torch.dot(a, b)
    
    # 如果向量几乎完全相反，添加一些随机噪声避免数值问题
    # If vectors are exactly opposite, we add a little noise to one of them
    if c < -1 + 1e-8:
        eps = (torch.rand(3) - 0.5) * 0.01
        return rotation_matrix(a + eps, b)
    
    # 计算叉积的范数（旋转轴的长度）
    s = torch.linalg.norm(v)
    
    # 构建旋转轴的斜对称矩阵
    skew_sym_mat = torch.Tensor(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ]
    )
    
    # 使用Rodrigues旋转公式计算旋转矩阵
    # R = I + sin(θ)K + (1-cos(θ))K²，其中K是旋转轴的斜对称矩阵
    return torch.eye(3) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * ((1 - c) / (s**2 + 1e-8))


def auto_orient_and_center_poses(
    poses, method="up", center_poses=True
):
    """Orients and centers the poses. We provide two methods for orientation: pca and up.

    pca: Orient the poses so that the principal component of the points is aligned with the axes.
        This method works well when all of the cameras are in the same plane.
    up: Orient the poses so that the average up vector is aligned with the z axis.
        This method works well when images are not at arbitrary angles.


    Args:
        poses: The poses to orient.
        method: The method to use for orientation.
        center_poses: If True, the poses are centered around the origin.

    Returns:
        The oriented poses.
    """
    # 自动调整相机姿态的方向和中心位置
    # 提供两种方法：pca和up
    # pca：使点的主成分与坐标轴对齐，适用于所有相机在同一平面上的情况
    # up：使平均上向量与z轴对齐，适用于图像角度不任意的情况

    # 提取平移部分（poses矩阵的最后一列）
    translation = poses[..., :3, 3]

    # 计算平均平移向量
    mean_translation = torch.mean(translation, dim=0)
    
    # 计算每个平移向量与平均平移向量的差值
    translation_diff = translation - mean_translation

    # 根据center_poses参数决定是否将姿态中心化
    if center_poses:
        translation = mean_translation
    else:
        translation = torch.zeros_like(mean_translation)

    if method == "pca":
        # PCA方法：使用主成分分析确定方向
        
        # 计算协方差矩阵的特征向量
        _, eigvec = torch.linalg.eigh(translation_diff.T @ translation_diff)
        
        # 翻转特征向量顺序（从大到小）
        eigvec = torch.flip(eigvec, dims=(-1,))

        # 确保右手坐标系（行列式为正）
        if torch.linalg.det(eigvec) < 0:
            eigvec[:, 2] = -eigvec[:, 2]

        # 构建变换矩阵：[R|t]，其中R是旋转矩阵，t是平移向量
        transform = torch.cat([eigvec, eigvec @ -translation[..., None]], dim=-1)
        
        # 应用变换到原始姿态
        oriented_poses = transform @ poses

        # 进一步调整，确保平均姿态的某些元素符合预期方向
        if oriented_poses.mean(axis=0)[2, 1] < 0:
            oriented_poses[:, 1:3] = -1 * oriented_poses[:, 1:3]
            
    elif method == "up":
        # UP方法：使平均上向量与z轴对齐
        
        # 计算所有姿态的平均上向量（假设y轴为上）
        up = torch.mean(poses[:, :3, 1], dim=0)
        
        # 归一化上向量
        up = up / torch.linalg.norm(up)

        # 计算将平均上向量旋转到z轴的旋转矩阵
        rotation = rotation_matrix(up, torch.Tensor([0, 0, 1]))
        
        # 构建完整的变换矩阵
        transform = torch.cat([rotation, rotation @ -translation[..., None]], dim=-1)
        
        # 应用变换到原始姿态
        oriented_poses = transform @ poses
        
    elif method == "none":
        # 不进行方向调整，只进行平移
        transform = torch.eye(4)
        transform[:3, 3] = -translation
        transform = transform[:3, :]
        oriented_poses = transform @ poses

    # 返回调整后的姿态和变换矩阵
    return oriented_poses, transform


