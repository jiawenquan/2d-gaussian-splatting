# adapted from https://github.com/jzhangbs/DTUeval-python
import numpy as np
import open3d as o3d
import sklearn.neighbors as skln
from tqdm import tqdm
from scipy.io import loadmat
import multiprocessing as mp
import argparse

def sample_single_tri(input_):
    """
    对单个三角形进行采样
    
    参数:
        input_: 包含三角形参数的元组 (n1, n2, v1, v2, tri_vert)
            n1, n2: 采样网格的大小
            v1, v2: 三角形的两个边向量
            tri_vert: 三角形的一个顶点
            
    返回:
        三角形上采样的点坐标
    """
    n1, n2, v1, v2, tri_vert = input_
    # 创建网格坐标
    c = np.mgrid[:n1+1, :n2+1]
    c += 0.5  # 中心偏移
    # 归一化网格坐标
    c[0] /= max(n1, 1e-7)
    c[1] /= max(n2, 1e-7)
    c = np.transpose(c, (1,2,0))
    # 选择三角形内的点(坐标和小于1的点)
    k = c[c.sum(axis=-1) < 1]  # m2
    # 根据重心坐标计算三角形内的点
    q = v1 * k[:,:1] + v2 * k[:,1:] + tri_vert
    return q

def write_vis_pcd(file, points, colors):
    """
    将点云和颜色写入文件
    
    参数:
        file: 输出文件路径
        points: 点云坐标
        colors: 点云颜色
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(file, pcd)

if __name__ == '__main__':
    mp.freeze_support()  # 支持Windows下的多进程

    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='data_in.ply')  # 输入模型路径
    parser.add_argument('--scan', type=int, default=1)  # 扫描ID
    parser.add_argument('--mode', type=str, default='mesh', choices=['mesh', 'pcd'])  # 输入模式，网格或点云
    parser.add_argument('--dataset_dir', type=str, default='.')  # 数据集目录
    parser.add_argument('--vis_out_dir', type=str, default='.')  # 可视化输出目录
    parser.add_argument('--downsample_density', type=float, default=0.2)  # 下采样密度
    parser.add_argument('--patch_size', type=float, default=60)  # 补丁大小
    parser.add_argument('--max_dist', type=float, default=20)  # 最大距离阈值
    parser.add_argument('--visualize_threshold', type=float, default=10)  # 可视化阈值
    args = parser.parse_args()

    thresh = args.downsample_density
    if args.mode == 'mesh':
        # 如果输入是网格模型
        pbar = tqdm(total=9)
        pbar.set_description('read data mesh')
        # 读取三角网格
        data_mesh = o3d.io.read_triangle_mesh(args.data)

        # 获取顶点和三角形
        vertices = np.asarray(data_mesh.vertices)
        triangles = np.asarray(data_mesh.triangles)
        tri_vert = vertices[triangles]  # 获取三角形的顶点坐标

        pbar.update(1)
        pbar.set_description('sample pcd from mesh')
        # 计算三角形的两条边
        v1 = tri_vert[:,1] - tri_vert[:,0]  # 边1
        v2 = tri_vert[:,2] - tri_vert[:,0]  # 边2
        # 计算边长
        l1 = np.linalg.norm(v1, axis=-1, keepdims=True)
        l2 = np.linalg.norm(v2, axis=-1, keepdims=True)
        # 计算三角形的面积(叉积的长度的一半)
        area2 = np.linalg.norm(np.cross(v1, v2), axis=-1, keepdims=True)
        # 过滤掉面积为零的三角形
        non_zero_area = (area2 > 0)[:,0]
        l1, l2, area2, v1, v2, tri_vert = [
            arr[non_zero_area] for arr in [l1, l2, area2, v1, v2, tri_vert]
        ]
        # 计算采样阈值
        thr = thresh * np.sqrt(l1 * l2 / area2)
        # 计算每条边上需要采样的点数
        n1 = np.floor(l1 / thr)
        n2 = np.floor(l2 / thr)

        # 使用多进程对每个三角形进行采样
        with mp.Pool() as mp_pool:
            new_pts = mp_pool.map(sample_single_tri, ((n1[i,0], n2[i,0], v1[i:i+1], v2[i:i+1], tri_vert[i:i+1,0]) for i in range(len(n1))), chunksize=1024)

        # 合并所有采样点
        new_pts = np.concatenate(new_pts, axis=0)
        data_pcd = np.concatenate([vertices, new_pts], axis=0)  # 原始顶点和采样点
    
    elif args.mode == 'pcd':
        # 如果输入是点云
        pbar = tqdm(total=8)
        pbar.set_description('read data pcd')
        data_pcd_o3d = o3d.io.read_point_cloud(args.data)
        data_pcd = np.asarray(data_pcd_o3d.points)

    pbar.update(1)
    pbar.set_description('random shuffle pcd index')
    # 随机打乱点云顺序
    shuffle_rng = np.random.default_rng()
    shuffle_rng.shuffle(data_pcd, axis=0)

    pbar.update(1)
    pbar.set_description('downsample pcd')
    # 使用最近邻算法进行下采样
    nn_engine = skln.NearestNeighbors(n_neighbors=1, radius=thresh, algorithm='kd_tree', n_jobs=-1)
    nn_engine.fit(data_pcd)
    # 查找每个点半径内的所有点
    rnn_idxs = nn_engine.radius_neighbors(data_pcd, radius=thresh, return_distance=False)
    # 创建掩码用于下采样
    mask = np.ones(data_pcd.shape[0], dtype=np.bool_)
    # 对于每个点，如果它被保留，则其邻居被移除
    for curr, idxs in enumerate(rnn_idxs):
        if mask[curr]:
            mask[idxs] = 0
            mask[curr] = 1
    # 应用掩码获取下采样的点云
    data_down = data_pcd[mask]

    pbar.update(1)
    pbar.set_description('masking data pcd')
    # 加载观察掩码和边界框
    obs_mask_file = loadmat(f'{args.dataset_dir}/ObsMask/ObsMask{args.scan}_10.mat')
    ObsMask, BB, Res = [obs_mask_file[attr] for attr in ['ObsMask', 'BB', 'Res']]
    BB = BB.astype(np.float32)

    # 添加边界扩展
    patch = args.patch_size
    # 判断点是否在扩展边界框内
    inbound = ((data_down >= BB[:1]-patch) & (data_down < BB[1:]+patch*2)).sum(axis=-1) ==3
    data_in = data_down[inbound]  # 保留在边界内的点

    # 将点坐标转换为网格索引
    data_grid = np.around((data_in - BB[:1]) / Res).astype(np.int32)
    # 判断网格索引是否在观察掩码范围内
    grid_inbound = ((data_grid >= 0) & (data_grid < np.expand_dims(ObsMask.shape, 0))).sum(axis=-1) ==3
    data_grid_in = data_grid[grid_inbound]
    # 获取在观察掩码内的点
    in_obs = ObsMask[data_grid_in[:,0], data_grid_in[:,1], data_grid_in[:,2]].astype(np.bool_)
    data_in_obs = data_in[grid_inbound][in_obs]  # 最终过滤后的点云

    pbar.update(1)
    pbar.set_description('read STL pcd')
    # 读取STL格式的点云（参考模型）
    stl_pcd = o3d.io.read_point_cloud(f'{args.dataset_dir}/Points/stl/stl{args.scan:03}_total.ply')
    stl = np.asarray(stl_pcd.points)

    pbar.update(1)
    pbar.set_description('compute data2stl')
    # 计算data到stl的距离（精度度量）
    nn_engine.fit(stl)
    dist_d2s, idx_d2s = nn_engine.kneighbors(data_in_obs, n_neighbors=1, return_distance=True)
    max_dist = args.max_dist
    # 计算平均距离，忽略大于最大距离的点
    mean_d2s = dist_d2s[dist_d2s < max_dist].mean()

    pbar.update(1)
    pbar.set_description('compute stl2data')
    # 加载地面平面参数
    ground_plane = loadmat(f'{args.dataset_dir}/ObsMask/Plane{args.scan}.mat')['P']

    # 将点转换为齐次坐标
    stl_hom = np.concatenate([stl, np.ones_like(stl[:,:1])], -1)
    # 判断点是否在地面平面上方
    above = (ground_plane.reshape((1,4)) * stl_hom).sum(-1) > 0
    stl_above = stl[above]  # 保留在地面上方的点

    # 计算stl到data的距离（完整度度量）
    nn_engine.fit(data_in)
    dist_s2d, idx_s2d = nn_engine.kneighbors(stl_above, n_neighbors=1, return_distance=True)
    # 计算平均距离，忽略大于最大距离的点
    mean_s2d = dist_s2d[dist_s2d < max_dist].mean()

    pbar.update(1)
    pbar.set_description('visualize error')
    # 可视化误差
    vis_dist = args.visualize_threshold
    # 定义颜色
    R = np.array([[1,0,0]], dtype=np.float64)  # 红色
    G = np.array([[0,1,0]], dtype=np.float64)  # 绿色
    B = np.array([[0,0,1]], dtype=np.float64)  # 蓝色
    W = np.array([[1,1,1]], dtype=np.float64)  # 白色
    
    # 为data点云着色
    data_color = np.tile(B, (data_down.shape[0], 1))  # 初始为蓝色
    data_alpha = dist_d2s.clip(max=vis_dist) / vis_dist  # 归一化距离作为透明度
    # 根据距离设置颜色：红色到白色的渐变
    data_color[ np.where(inbound)[0][grid_inbound][in_obs] ] = R * data_alpha + W * (1-data_alpha)
    # 距离过大的点设为绿色
    data_color[ np.where(inbound)[0][grid_inbound][in_obs][dist_d2s[:,0] >= max_dist] ] = G
    # 写入可视化文件
    write_vis_pcd(f'{args.vis_out_dir}/vis_{args.scan:03}_d2s.ply', data_down, data_color)
    
    # 为stl点云着色
    stl_color = np.tile(B, (stl.shape[0], 1))  # 初始为蓝色
    stl_alpha = dist_s2d.clip(max=vis_dist) / vis_dist  # 归一化距离作为透明度
    # 根据距离设置颜色：红色到白色的渐变
    stl_color[ np.where(above)[0] ] = R * stl_alpha + W * (1-stl_alpha)
    # 距离过大的点设为绿色
    stl_color[ np.where(above)[0][dist_s2d[:,0] >= max_dist] ] = G
    # 写入可视化文件
    write_vis_pcd(f'{args.vis_out_dir}/vis_{args.scan:03}_s2d.ply', stl, stl_color)

    pbar.update(1)
    pbar.set_description('done')
    pbar.close()
    # 计算总体评分（精度和完整度的平均值）
    over_all = (mean_d2s + mean_s2d) / 2
    print(mean_d2s, mean_s2d, over_all)
    
    # 将结果写入JSON文件
    import json
    with open(f'{args.vis_out_dir}/results.json', 'w') as fp:
        json.dump({
            'mean_d2s': mean_d2s,  # 精度（data到stl的平均距离）
            'mean_s2d': mean_s2d,  # 完整度（stl到data的平均距离）
            'overall': over_all,   # 总体评分
        }, fp, indent=True)