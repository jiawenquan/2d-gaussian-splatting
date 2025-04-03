import numpy as np
import open3d as o3d


class CameraPose:
    """
    相机位姿类，用于存储相机的元数据和位姿矩阵
    """
    def __init__(self, meta, mat):
        """
        初始化相机位姿对象
        
        参数:
            meta: 元数据（通常包含帧索引等信息）
            mat: 位姿矩阵（4x4变换矩阵）
        """
        self.metadata = meta
        self.pose = mat

    def __str__(self):
        """
        返回相机位姿的字符串表示
        """
        return ("Metadata : " + " ".join(map(str, self.metadata)) + "\n" +
                "Pose : " + "\n" + np.array_str(self.pose))


def convert_trajectory_to_pointcloud(traj):
    """
    将相机轨迹转换为点云
    
    参数:
        traj: 相机轨迹（CameraPose对象列表）
        
    返回:
        包含相机位置的点云对象
    
    该函数提取每个相机位姿的平移部分（位置）创建点云，
    用于可视化相机轨迹或进行轨迹配准。
    """
    pcd = o3d.geometry.PointCloud()
    for t in traj:
        pcd.points.append(t.pose[:3, 3])  # 提取位姿矩阵的平移部分（位置）
    return pcd


def read_trajectory(filename):
    """
    从文件读取相机轨迹
    
    参数:
        filename: 轨迹文件路径
        
    返回:
        CameraPose对象列表
    
    轨迹文件格式：
    - 每个相机位姿由多行组成
    - 第一行包含元数据（通常是帧索引）
    - 接下来4行是4x4位姿矩阵，每行对应矩阵的一行
    """
    traj = []
    with open(filename, "r") as f:
        metastr = f.readline()
        while metastr:
            metadata = map(int, metastr.split())  # 解析元数据
            mat = np.zeros(shape=(4, 4))  # 创建4x4位姿矩阵
            for i in range(4):
                matstr = f.readline()
                mat[i, :] = np.fromstring(matstr, dtype=float, sep=" \t")  # 解析矩阵行
            traj.append(CameraPose(metadata, mat))  # 添加相机位姿到轨迹
            metastr = f.readline()  # 读取下一个位姿的元数据
    return traj


def write_trajectory(traj, filename):
    """
    将相机轨迹写入文件
    
    参数:
        traj: CameraPose对象列表
        filename: 要写入的文件路径
    
    轨迹文件格式：
    - 每个相机位姿由多行组成
    - 第一行包含元数据（通常是帧索引）
    - 接下来4行是4x4位姿矩阵，每行对应矩阵的一行
    - 使用高精度浮点数（12位小数）保存位姿信息
    """
    with open(filename, "w") as f:
        for x in traj:
            p = x.pose.tolist()  # 将位姿矩阵转换为列表
            f.write(" ".join(map(str, x.metadata)) + "\n")  # 写入元数据
            f.write("\n".join(
                " ".join(map("{0:.12f}".format, p[i])) for i in range(4)))  # 写入位姿矩阵（高精度）
            f.write("\n")
