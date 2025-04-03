# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import enum
import types
from typing import List, Mapping, Optional, Text, Tuple, Union
import copy
from PIL import Image
import mediapy as media
from matplotlib import cm
from tqdm import tqdm

import torch

def normalize(x: np.ndarray) -> np.ndarray:
  """Normalization helper function."""
  return x / np.linalg.norm(x)  # 将向量归一化，使其范数为1

def pad_poses(p: np.ndarray) -> np.ndarray:
  """Pad [..., 3, 4] pose matrices with a homogeneous bottom row [0,0,0,1]."""
  bottom = np.broadcast_to([0, 0, 0, 1.], p[..., :1, :4].shape)  # 创建底行 [0,0,0,1]
  return np.concatenate([p[..., :3, :4], bottom], axis=-2)  # 将底行与原始姿态矩阵连接


def unpad_poses(p: np.ndarray) -> np.ndarray:
  """Remove the homogeneous bottom row from [..., 4, 4] pose matrices."""
  return p[..., :3, :4]  # 移除底行，只保留3×4部分


def recenter_poses(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Recenter poses around the origin."""
  cam2world = average_pose(poses)  # 计算平均姿态
  transform = np.linalg.inv(pad_poses(cam2world))  # 计算从平均姿态到原点的变换
  poses = transform @ pad_poses(poses)  # 应用变换到所有姿态
  return unpad_poses(poses), transform  # 返回变换后的姿态和变换矩阵


def average_pose(poses: np.ndarray) -> np.ndarray:
  """New pose using average position, z-axis, and up vector of input poses."""
  position = poses[:, :3, 3].mean(0)  # 计算平均位置
  z_axis = poses[:, :3, 2].mean(0)  # 计算平均z轴（观察方向）
  up = poses[:, :3, 1].mean(0)  # 计算平均上向量
  cam2world = viewmatrix(z_axis, up, position)  # 使用平均参数创建视图矩阵
  return cam2world

def viewmatrix(lookdir: np.ndarray, up: np.ndarray,
               position: np.ndarray) -> np.ndarray:
  """Construct lookat view matrix."""
  vec2 = normalize(lookdir)  # 归一化观察方向作为z轴
  vec0 = normalize(np.cross(up, vec2))  # 计算x轴：上向量与z轴的叉积，并归一化
  vec1 = normalize(np.cross(vec2, vec0))  # 计算y轴：z轴与x轴的叉积，并归一化
  m = np.stack([vec0, vec1, vec2, position], axis=1)  # 堆叠向量和位置，构建视图矩阵
  return m

def focus_point_fn(poses: np.ndarray) -> np.ndarray:
  """Calculate nearest point to all focal axes in poses."""
  directions, origins = poses[:, :3, 2:3], poses[:, :3, 3:4]  # 提取观察方向和相机原点
  # 构建最小二乘法求解焦点的矩阵
  m = np.eye(3) - directions * np.transpose(directions, [0, 2, 1])  # 计算投影矩阵 I - d*d^T
  mt_m = np.transpose(m, [0, 2, 1]) @ m  # 计算矩阵乘积 M^T * M
  # 求解线性方程组，找到最接近所有视线的点
  focus_pt = np.linalg.inv(mt_m.mean(0)) @ (mt_m @ origins).mean(0)[:, 0]
  return focus_pt

def transform_poses_pca(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
  """Transforms poses so principal components lie on XYZ axes.

  Args:
    poses: a (N, 3, 4) array containing the cameras' camera to world transforms.

  Returns:
    A tuple (poses, transform), with the transformed poses and the applied
    camera_to_world transforms.
  """
  t = poses[:, :3, 3]  # 提取相机位置
  t_mean = t.mean(axis=0)  # 计算平均位置
  t = t - t_mean  # 中心化数据

  # 计算主成分（PCA）
  eigval, eigvec = np.linalg.eig(t.T @ t)  # 计算协方差矩阵的特征值和特征向量
  # 按特征值从大到小排序特征向量
  inds = np.argsort(eigval)[::-1]
  eigvec = eigvec[:, inds]
  rot = eigvec.T  # 旋转矩阵
  # 确保旋转矩阵是右手坐标系（行列式为正）
  if np.linalg.det(rot) < 0:
    rot = np.diag(np.array([1, 1, -1])) @ rot

  # 构建变换矩阵，包括旋转和平移
  transform = np.concatenate([rot, rot @ -t_mean[:, None]], -1)
  poses_recentered = unpad_poses(transform @ pad_poses(poses))  # 应用变换到姿态
  transform = np.concatenate([transform, np.eye(4)[3:]], axis=0)  # 补全变换矩阵

  # 如果y轴的z分量为负，翻转坐标系
  if poses_recentered.mean(axis=0)[2, 1] < 0:
    poses_recentered = np.diag(np.array([1, -1, -1])) @ poses_recentered
    transform = np.diag(np.array([1, -1, -1, 1])) @ transform

  return poses_recentered, transform
  # points = np.random.rand(3,100)
  # points_h = np.concatenate((points,np.ones_like(points[:1])), axis=0)
  # (poses_recentered @ points_h)[0]
  # (transform @ pad_poses(poses) @ points_h)[0,:3]
  # import pdb; pdb.set_trace()

  # # Just make sure it's it in the [-1, 1]^3 cube
  # scale_factor = 1. / np.max(np.abs(poses_recentered[:, :3, 3]))
  # poses_recentered[:, :3, 3] *= scale_factor
  # transform = np.diag(np.array([scale_factor] * 3 + [1])) @ transform

  # return poses_recentered, transform

def generate_ellipse_path(poses: np.ndarray,
                          n_frames: int = 120,
                          const_speed: bool = True,
                          z_variation: float = 0.,
                          z_phase: float = 0.) -> np.ndarray:
  """Generate an elliptical render path based on the given poses."""
  # Calculate the focal point for the path (cameras point toward this).
  center = focus_point_fn(poses)  # 计算焦点，即场景中心
  # Path height sits at z=0 (in middle of zero-mean capture pattern).
  offset = np.array([center[0], center[1], 0])  # 创建偏移量，z设为0，使路径高度居中

  # Calculate scaling for ellipse axes based on input camera positions.
  sc = np.percentile(np.abs(poses[:, :3, 3] - offset), 90, axis=0)  # 计算椭圆轴的缩放比例
  # Use ellipse that is symmetric about the focal point in xy.
  low = -sc + offset  # 椭圆边界的低点
  high = sc + offset  # 椭圆边界的高点
  # Optional height variation need not be symmetric
  z_low = np.percentile((poses[:, :3, 3]), 10, axis=0)  # z轴的低点（10%分位数）
  z_high = np.percentile((poses[:, :3, 3]), 90, axis=0)  # z轴的高点（90%分位数）

  def get_positions(theta):
    # Interpolate between bounds with trig functions to get ellipse in x-y.
    # Optionally also interpolate in z to change camera height along path.
    return np.stack([
        low[0] + (high - low)[0] * (np.cos(theta) * .5 + .5),  # x坐标：余弦插值
        low[1] + (high - low)[1] * (np.sin(theta) * .5 + .5),  # y坐标：正弦插值
        z_variation * (z_low[2] + (z_high - z_low)[2] *  # z坐标：可选的高度变化
                       (np.cos(theta + 2 * np.pi * z_phase) * .5 + .5)),
    ], -1)

  theta = np.linspace(0, 2. * np.pi, n_frames + 1, endpoint=True)  # 生成均匀分布的角度
  positions = get_positions(theta)  # 计算位置

  #if const_speed:

  # # Resample theta angles so that the velocity is closer to constant.
  # lengths = np.linalg.norm(positions[1:] - positions[:-1], axis=-1)
  # theta = stepfun.sample(None, theta, np.log(lengths), n_frames + 1)
  # positions = get_positions(theta)

  # Throw away duplicated last position.
  positions = positions[:-1]  # 丢弃重复的最后一个位置

  # Set path's up vector to axis closest to average of input pose up vectors.
  avg_up = poses[:, :3, 1].mean(0)  # 计算平均上向量
  avg_up = avg_up / np.linalg.norm(avg_up)  # 归一化上向量
  ind_up = np.argmax(np.abs(avg_up))  # 找到最接近上向量的坐标轴
  up = np.eye(3)[ind_up] * np.sign(avg_up[ind_up])  # 将上向量设置为该坐标轴

  return np.stack([viewmatrix(p - center, up, p) for p in positions])  # 为每个位置创建视图矩阵


def generate_path(viewpoint_cameras, n_frames=480):
  """
  根据给定的相机视点生成一个椭圆形渲染路径
  
  参数:
  viewpoint_cameras - 相机视点列表
  n_frames - 路径中的帧数
  
  返回:
  traj - 新的相机轨迹
  """
  # 将相机的世界到视图变换转换为相机到世界变换（c2w）
  c2ws = np.array([np.linalg.inv(np.asarray((cam.world_view_transform.T).cpu().numpy())) for cam in viewpoint_cameras])
  pose = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])  # 应用坐标系变换
  # 进行PCA变换，重新对齐坐标系
  pose_recenter, colmap_to_world_transform = transform_poses_pca(pose)

  # 生成新的椭圆形路径
  new_poses = generate_ellipse_path(poses=pose_recenter, n_frames=n_frames)
  # 将路径变换回原始坐标系
  new_poses = np.linalg.inv(colmap_to_world_transform) @ pad_poses(new_poses)

  traj = []  # 创建新的轨迹
  for c2w in new_poses:
      c2w = c2w @ np.diag([1, -1, -1, 1])  # 应用坐标系变换
      cam = copy.deepcopy(viewpoint_cameras[0])  # 复制第一个相机的参数
      # 确保图像尺寸是偶数
      cam.image_height = int(cam.image_height / 2) * 2
      cam.image_width = int(cam.image_width / 2) * 2
      # 设置新的相机参数
      cam.world_view_transform = torch.from_numpy(np.linalg.inv(c2w).T).float().cuda()  # 设置世界到视图变换
      cam.full_proj_transform = (cam.world_view_transform.unsqueeze(0).bmm(cam.projection_matrix.unsqueeze(0))).squeeze(0)  # 计算完整投影矩阵
      cam.camera_center = cam.world_view_transform.inverse()[3, :3]  # 计算相机中心
      traj.append(cam)

  return traj

def load_img(pth: str) -> np.ndarray:
  """Load an image and cast to float32."""
  with open(pth, 'rb') as f:
    image = np.array(Image.open(f), dtype=np.float32)  # 打开图像并转换为float32类型的NumPy数组
  return image


def create_videos(base_dir, input_dir, out_name, num_frames=480):
  """Creates videos out of the images saved to disk."""
  # Last two parts of checkpoint path are experiment name and scene name.
  video_prefix = f'{out_name}'  # 视频文件名前缀

  zpad = max(5, len(str(num_frames - 1)))  # 计算帧索引的零填充长度
  idx_to_str = lambda idx: str(idx).zfill(zpad)  # 将索引转换为零填充的字符串

  os.makedirs(base_dir, exist_ok=True)  # 创建输出目录
  render_dist_curve_fn = np.log  # 用于渲染深度的曲线函数
  
  # Load one example frame to get image shape and depth range.
  depth_file = os.path.join(input_dir, 'vis', f'depth_{idx_to_str(0)}.tiff')  # 第一帧深度文件路径
  depth_frame = load_img(depth_file)  # 加载深度图
  shape = depth_frame.shape  # 获取图像形状
  p = 3  # 百分位数阈值
  distance_limits = np.percentile(depth_frame.flatten(), [p, 100 - p])  # 计算深度范围
  lo, hi = [render_dist_curve_fn(x) for x in distance_limits]  # 应用对数变换
  print(f'Video shape is {shape[:2]}')

  # 设置视频参数
  video_kwargs = {
      'shape': shape[:2],  # 视频形状
      'codec': 'h264',  # 编码器
      'fps': 60,  # 帧率
      'crf': 18,  # 压缩质量
  }
  
  # 为不同类型的数据创建视频
  for k in ['depth', 'normal', 'color']:
    video_file = os.path.join(base_dir, f'{video_prefix}_{k}.mp4')  # 输出视频文件路径
    input_format = 'gray' if k == 'alpha' else 'rgb'  # 输入格式
    

    file_ext = 'png' if k in ['color', 'normal'] else 'tiff'  # 文件扩展名
    idx = 0  # 初始帧索引

    # 检查文件是否存在
    if k == 'color':
      file0 = os.path.join(input_dir, 'renders', f'{idx_to_str(0)}.{file_ext}')
    else:
      file0 = os.path.join(input_dir, 'vis', f'{k}_{idx_to_str(0)}.{file_ext}')

    if not os.path.exists(file0):
      print(f'Images missing for tag {k}')
      continue
    print(f'Making video {video_file}...')
    # 使用mediapy创建视频
    with media.VideoWriter(
        video_file, **video_kwargs, input_format=input_format) as writer:
      for idx in tqdm(range(num_frames)):
        # 构建输入图像路径
        if k == 'color':
          img_file = os.path.join(input_dir, 'renders', f'{idx_to_str(idx)}.{file_ext}')
        else:
          img_file = os.path.join(input_dir, 'vis', f'{k}_{idx_to_str(idx)}.{file_ext}')

        if not os.path.exists(img_file):
          ValueError(f'Image file {img_file} does not exist.')
        img = load_img(img_file)  # 加载图像
        if k in ['color', 'normal']:
          img = img / 255.  # 归一化
        elif k.startswith('depth'):
          img = render_dist_curve_fn(img)  # 应用深度变换
          img = np.clip((img - np.minimum(lo, hi)) / np.abs(hi - lo), 0, 1)  # 归一化到[0,1]
          img = cm.get_cmap('turbo')(img)[..., :3]  # 应用颜色映射

        frame = (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)  # 转换为uint8
        writer.add_image(frame)  # 添加帧到视频
        idx += 1

def save_img_u8(img, pth):
  """Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
  with open(pth, 'wb') as f:
    # 将图像裁剪到[0,1]范围，转换为uint8，并保存为PNG
    Image.fromarray(
        (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(np.uint8)).save(
            f, 'PNG')


def save_img_f32(depthmap, pth):
  """Save an image (probably a depthmap) to disk as a float32 TIFF."""
  with open(pth, 'wb') as f:
    # 将深度图转换为float32，并保存为TIFF
    Image.fromarray(np.nan_to_num(depthmap).astype(np.float32)).save(f, 'TIFF')