#
# Copyright (C) 2024, ShanghaiTech
# SVIP research group, https://github.com/svip-lab
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  huangbb@shanghaitech.edu.cn
#

import numpy as np
import torch
import trimesh
from skimage import measure

# 带有收缩变换的Marching Cubes算法，用于从SDF（符号距离场）生成3D网格
# 修改自：https://github.com/autonomousvision/sdfstudio/blob/370902a10dbef08cb3fe4391bd3ed1e227b5c165/nerfstudio/utils/marching_cubes.py#L201
def marching_cubes_with_contraction(
    sdf,  # SDF函数，输入点坐标，输出SDF值
    resolution=512,  # 体素网格的分辨率
    bounding_box_min=(-1.0, -1.0, -1.0),  # 包围盒最小点坐标
    bounding_box_max=(1.0, 1.0, 1.0),     # 包围盒最大点坐标
    return_mesh=False,  # 是否返回网格对象
    level=0,            # 等值面的值，通常为0（表示表面）
    simplify_mesh=True, # 是否简化网格
    inv_contraction=None,  # 逆收缩函数，用于将收缩空间中的点映射回原始空间
    max_range=32.0,     # 裁剪范围
):
    # 确保分辨率是512的倍数（为分块处理做准备）
    assert resolution % 512 == 0

    resN = resolution  # 总体分辨率
    cropN = 512        # 每个块的分辨率
    level = 0          # 等值面的值
    N = resN // cropN  # 每个维度上的块数

    # 设置网格的边界
    grid_min = bounding_box_min
    grid_max = bounding_box_max
    # 在每个维度上创建均匀的坐标点
    xs = np.linspace(grid_min[0], grid_max[0], N + 1)
    ys = np.linspace(grid_min[1], grid_max[1], N + 1)
    zs = np.linspace(grid_min[2], grid_max[2], N + 1)

    # 存储各块生成的网格
    meshes = []
    # 逐块处理，分割大规模体素网格以节省内存
    for i in range(N):
        for j in range(N):
            for k in range(N):
                print(i, j, k)
                # 当前块的范围
                x_min, x_max = xs[i], xs[i + 1]
                y_min, y_max = ys[j], ys[j + 1]
                z_min, z_max = zs[k], zs[k + 1]

                # 在当前块内生成均匀的坐标点
                x = torch.linspace(x_min, x_max, cropN).cuda()
                y = torch.linspace(y_min, y_max, cropN).cuda()
                z = torch.linspace(z_min, z_max, cropN).cuda()

                # 创建三维网格点
                xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
                points = torch.tensor(torch.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float).cuda()

                # 定义评估函数，分批处理点以避免内存溢出
                @torch.no_grad()
                def evaluate(points):
                    z = []
                    for _, pnts in enumerate(torch.split(points, 256**3, dim=0)):
                        z.append(sdf(pnts))
                    z = torch.cat(z, axis=0)
                    return z

                # 构建点金字塔
                points = points.reshape(cropN, cropN, cropN, 3)
                points = points.reshape(-1, 3)
                # 计算每个点的SDF值
                pts_sdf = evaluate(points.contiguous())
                z = pts_sdf.detach().cpu().numpy()
                
                # 如果当前块内的SDF值不包含0（即没有表面穿过），则跳过
                if not (np.min(z) > level or np.max(z) < level):
                    z = z.astype(np.float32)
                    # 使用Marching Cubes算法提取等值面
                    verts, faces, normals, _ = measure.marching_cubes(
                        volume=z.reshape(cropN, cropN, cropN),
                        level=level,
                        spacing=(
                            (x_max - x_min) / (cropN - 1),
                            (y_max - y_min) / (cropN - 1),
                            (z_max - z_min) / (cropN - 1),
                        ),
                    )
                    # 将顶点坐标转换回原始空间
                    verts = verts + np.array([x_min, y_min, z_min])
                    # 创建网格对象
                    meshcrop = trimesh.Trimesh(verts, faces, normals)
                    meshes.append(meshcrop)
                
                print("finished one block")

    # 合并所有块的网格
    combined = trimesh.util.concatenate(meshes)
    # 合并重复的顶点
    combined.merge_vertices(digits_vertex=6)

    # 如果提供了逆收缩函数，将顶点从收缩空间映射回原始空间
    if inv_contraction is not None:
        combined.vertices = inv_contraction(torch.from_numpy(combined.vertices).float().cuda()).cpu().numpy()
        # 裁剪顶点到指定范围内
        combined.vertices = np.clip(combined.vertices, -max_range, max_range)
    
    return combined