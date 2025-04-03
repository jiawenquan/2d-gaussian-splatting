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

# ===== 导入电脑需要的工具箱 =====
import torch  # PyTorch深度学习库，就像电脑的"思考"工具
import numpy as np  # 数值计算库，就像电脑的"计算器"
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation  # 一些特殊的数学工具
from torch import nn  # PyTorch神经网络模块，就像电脑的"神经系统"
import os  # 操作系统接口，帮助电脑找到文件
from utils.system_utils import mkdir_p  # 创建文件夹的工具，就像帮电脑整理柜子
from plyfile import PlyData, PlyElement  # 处理3D模型文件的工具
from utils.sh_utils import RGB2SH  # 颜色转换工具，把普通颜色变成特殊格式
from simple_knn._C import distCUDA2  # 快速计算点之间距离的工具
from utils.graphics_utils import BasicPointCloud  # 基础点云工具，处理一堆3D点
from utils.general_utils import strip_symmetric, build_scaling_rotation  # 更多数学工具

class GaussianModel:
    """
    # ===== 高斯点模型：3D场景的小星星 =====
    想象一下，这个类就像一个装满亮晶晶小星星的盒子！
    每个小星星（高斯点）都有：
    - 位置：在3D空间的哪里
    - 颜色：是什么颜色的小星星
    - 大小：有多大
    - 方向：朝哪个方向
    - 透明度：有多透明
    
    所有这些小星星一起，能画出漂亮的3D场景！
    """

    def setup_functions(self):
        """
        # ===== 设置魔法变形函数 =====
        这些函数就像魔法咒语，可以把数字变成我们想要的样子。
        
        想象一下：
        - 缩放激活函数是把"种子"变成真正的"花朵大小"
        - 不透明度激活函数是把任意数字变成0到1之间（0是完全透明，1是完全不透明）
        - 旋转激活函数确保方向是正确的
        """
        def build_covariance_from_scaling_rotation(center, scaling, scaling_modifier, rotation):
            """
            # 创建椭圆形状信息
            想象这个函数是教小星星如何在空间中"伸展"：
            - center：小星星的位置
            - scaling：小星星要伸展多大
            - rotation：小星星要往哪个方向伸展
            最后返回的是小星星的完整"伸展指南"
            """
            RS = build_scaling_rotation(torch.cat([scaling * scaling_modifier, torch.ones_like(scaling)], dim=-1), rotation).permute(0,2,1)
            trans = torch.zeros((center.shape[0], 4, 4), dtype=torch.float, device="cuda")
            trans[:,:3,:3] = RS
            trans[:, 3,:3] = center
            trans[:, 3, 3] = 1
            return trans
        
        # === 缩放魔法 ===
        # 用指数函数把任何数字变成正数（因为大小不能是负的！）
        self.scaling_activation = torch.exp  # 把数字变大（像吹气球）
        self.scaling_inverse_activation = torch.log  # 反向操作（像放气）

        # === 其他魔法 ===
        self.covariance_activation = build_covariance_from_scaling_rotation  # 教小星星如何伸展
        self.opacity_activation = torch.sigmoid  # 把任何数字变成0-1之间（透明度）
        self.inverse_opacity_activation = inverse_sigmoid  # 反向操作
        self.rotation_activation = torch.nn.functional.normalize  # 确保方向是规范的

    def __init__(self, sh_degree : int):
        """
        # ===== 创建一个新的小星星盒子 =====
        就像准备一个空盒子，等待装入漂亮的小星星。
        
        参数:
            sh_degree: 颜色的复杂程度（越大颜色越丰富）
            就像画笔可以有不同数量的颜色，数字越大，颜色越多样！
        """
        # === 颜色复杂度设置 ===
        self.active_sh_degree = 0  # 当前使用的颜色复杂度（从简单开始）
        self.max_sh_degree = sh_degree  # 最大能达到的颜色复杂度
        
        # === 准备空盒子，等待装入小星星的各种属性 ===
        self._xyz = torch.empty(0)  # 位置：小星星在哪里
        self._features_dc = torch.empty(0)  # 基础颜色：小星星的主要颜色
        self._features_rest = torch.empty(0)  # 额外颜色细节：让颜色更丰富
        self._scaling = torch.empty(0)  # 大小：小星星有多大
        self._rotation = torch.empty(0)  # 方向：小星星朝哪个方向
        self._opacity = torch.empty(0)  # 透明度：小星星有多透明
        
        # === 训练相关的空盒子 ===
        self.max_radii2D = torch.empty(0)  # 屏幕上的最大半径
        self.xyz_gradient_accum = torch.empty(0)  # 位置变化的累积记录
        self.denom = torch.empty(0)  # 计数器
        self.optimizer = None  # 优化器（帮助小星星变得更好）
        self.percent_dense = 0  # 密集程度
        self.spatial_lr_scale = 0  # 学习率缩放
        
        # 设置魔法变形函数
        self.setup_functions()  # 准备所有的魔法咒语

    def capture(self):
        """
        # ===== 给小星星盒子拍照 =====
        记录当前盒子里所有小星星的状态，就像拍一张照片，
        这样以后可以恢复到现在的样子。
        
        就像游戏中的"保存进度"功能！
        """
        return (
            self.active_sh_degree,  # 当前的颜色复杂度
            self._xyz,  # 所有小星星的位置
            self._features_dc,  # 基础颜色
            self._features_rest,  # 颜色细节
            self._scaling,  # 大小
            self._rotation,  # 方向
            self._opacity,  # 透明度
            self.max_radii2D,  # 屏幕最大半径
            self.xyz_gradient_accum,  # 位置变化记录
            self.denom,  # 计数器
            self.optimizer.state_dict(),  # 优化器状态
            self.spatial_lr_scale,  # 学习率缩放
        )
    
    def restore(self, model_args, training_args):
        """
        # ===== 从照片中恢复小星星盒子 =====
        使用之前"拍的照片"恢复小星星盒子的状态，
        就像游戏中的"读取存档"功能！
        
        参数:
            model_args: 之前保存的所有小星星信息
            training_args: 训练设置
        """
        # 依次恢复所有保存的小星星属性
        (self.active_sh_degree,  # 颜色复杂度
        self._xyz,  # 位置
        self._features_dc,  # 基础颜色
        self._features_rest,  # 颜色细节
        self._scaling,  # 大小
        self._rotation,  # 方向
        self._opacity,  # 透明度
        self.max_radii2D,  # 屏幕最大半径
        xyz_gradient_accum,  # 位置变化记录
        denom,  # 计数器
        opt_dict,  # 优化器状态
        self.spatial_lr_scale) = model_args
        
        # 设置训练环境
        self.training_setup(training_args)
        
        # 恢复其他状态
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)  # 恢复优化器

    @property
    def get_scaling(self):
        """
        # ===== 获取小星星的真实大小 =====
        使用魔法函数把存储的数字变成实际的大小
        就像把种子变成真正的花朵大小
        """
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        """
        # ===== 获取小星星的真实方向 =====
        使用魔法函数确保方向是正确的
        就像确保指南针指向正确的方向
        """
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        """
        # ===== 获取小星星的位置 =====
        返回所有小星星在3D空间中的位置
        就像知道每个小星星挂在天空的哪个位置
        """
        return self._xyz
    
    @property
    def get_features(self):
        """
        # ===== 获取小星星的完整颜色信息 =====
        组合基础颜色和细节，得到完整的颜色
        就像把简单的颜色和复杂的花纹组合起来
        """
        features_dc = self._features_dc  # 基础颜色（像蓝色、红色）
        features_rest = self._features_rest  # 细节（像花纹、渐变）
        return torch.cat((features_dc, features_rest), dim=1)  # 组合在一起
    
    @property
    def get_opacity(self):
        """
        # ===== 获取小星星的透明度 =====
        使用魔法函数把存储的数字变成0-1之间的透明度
        0是完全透明（看不见），1是完全不透明（很明显）
        就像魔法水晶球的清晰度
        """
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        """
        # ===== 获取小星星的形状信息 =====
        计算小星星在空间中如何"伸展"
        有点像计算气球的形状和方向
        
        参数:
            scaling_modifier: 大小调整系数，默认为1
            就像气球打气的多少
        """
        return self.covariance_activation(self.get_xyz, self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        """
        # ===== 提升颜色的复杂度 =====
        如果还没达到最高复杂度，就提升一级
        就像从8色蜡笔变成16色，再变成32色...
        让小星星的颜色表现更丰富！
        """
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1  # 颜色复杂度+1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        """
        # ===== 从一堆点创建小星星 =====
        把普通的3D点变成闪亮的小星星！
        就像把一把普通的沙子变成一盒闪亮的宝石。
        
        参数:
            pcd: 普通的3D点云（位置和颜色的集合）
            spatial_lr_scale: 位置学习速度的调整系数
        """
        self.spatial_lr_scale = spatial_lr_scale
        # 把点云数据搬到GPU上（让电脑算得更快）
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        # 把普通RGB颜色变成特殊的球谐系数格式
        # 就像把"红色"翻译成"特殊的颜色语言"
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        
        # 创建颜色特征容器（有基础颜色和细节两部分）
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color  # 设置基础颜色
        features[:, 3:, 1:] = 0.0  # 细节部分先设为0（以后会学习）

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # 计算点之间的距离，用来决定每个小星星的初始大小
        # 就像根据豆子的分布密度决定种出多大的花
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 2)
        
        # 随机初始化每个小星星的方向
        # 就像每个小星星随机朝向不同方向
        rots = torch.rand((fused_point_cloud.shape[0], 4), device="cuda")

        # 初始化每个小星星的透明度为0.1（有点透明）
        # 就像刚开始所有小星星都是朦胧的
        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        # 所有参数设为"可学习"的，这样它们可以在训练中变得更好
        # 就像给所有小星星设置"可以成长"的属性
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        
        # 初始化每个小星星在屏幕上的最大半径（暂时都是0）
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        """
        # ===== 设置小星星的训练计划 =====
        准备训练过程中需要的一切工具，就像准备上学前的书包。
        
        这个函数就像给小星星们制定"成长计划"：
        - 哪些属性要快速学习
        - 哪些属性要慢慢调整
        - 如何随着时间调整学习速度
        
        参数:
            training_args: 包含所有训练设置的对象
        """
        self.percent_dense = training_args.percent_dense
        
        # 初始化记录位置变化的计数器
        # 就像给每个小星星一个"移动日记"
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        # 为每个属性设置不同的学习速度
        # 就像不同科目有不同的学习计划
        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},  # 位置学习
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},  # 基础颜色学习
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},  # 颜色细节学习（慢一点）
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},  # 透明度学习
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},  # 大小学习
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}  # 方向学习
        ]

        # 创建Adam优化器（帮助学习的智能老师）
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        # 设置位置学习速度的变化规则
        # 就像随着时间调整学习进度，开始快一点，后来慢一点
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        '''
        # ===== 更新学习速度 =====
        根据当前训练进度调整学习速度
        就像根据学习阶段调整学习计划：
        - 刚开始学习快一些
        - 到后期调整变得更精细、更慢
        
        参数:
            iteration: 当前训练步数（第几课时）
        返回:
            更新后的位置学习速度
        '''
        # 遍历所有参数组，找到位置参数组并更新它的学习速度
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)  # 计算新的学习速度
                param_group['lr'] = lr  # 设置新的学习速度
                return lr  # 返回更新后的学习速度

    def construct_list_of_attributes(self):
        """
        # ===== 列出所有要保存的小星星属性 =====
        准备一个清单，列出保存小星星时需要记录的所有信息
        就像整理行李清单，确保不会遗漏任何重要物品
        
        返回:
            所有属性名称的清单
        """
        # 从基础的位置和方向开始
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']  # 位置(xyz)和法线(nxyz)
        
        # 添加所有基础颜色特征
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
            
        # 添加所有颜色细节特征
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
            
        # 添加透明度
        l.append('opacity')
        
        # 添加所有缩放（大小）属性
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
            
        # 添加所有旋转（方向）属性
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
            
        return l  # 返回完整的属性清单

    def save_ply(self, path):
        """
        # ===== 保存小星星到文件 =====
        把所有小星星的信息保存到一个特殊的3D文件（PLY格式）
        就像把所有小星星装进一个魔法盒子，以后可以再拿出来
        
        参数:
            path: 保存的文件路径（魔法盒子放在哪里）
        """
        # 确保保存的文件夹存在
        # 就像确保有个柜子可以放魔法盒子
        mkdir_p(os.path.dirname(path))

        # 获取所有参数的CPU版本（从GPU搬回来）
        # 就像把小星星从特殊容器中取出，准备装箱
        xyz = self._xyz.detach().cpu().numpy()  # 位置
        normals = np.zeros_like(xyz)  # 法线（都是0）
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()  # 基础颜色
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()  # 颜色细节
        opacities = self._opacity.detach().cpu().numpy()  # 透明度
        scale = self._scaling.detach().cpu().numpy()  # 大小
        rotation = self._rotation.detach().cpu().numpy()  # 方向

        # 创建PLY文件的数据类型（告诉文件每种属性怎么存）
        # 就像为魔法盒子里的每种物品设计专门的格子
        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        # 合并所有属性并写入PLY文件
        # 就像把所有物品按顺序整齐地放进魔法盒子
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)  # 最终写入文件（关上魔法盒子）

    def reset_opacity(self):
        """
        # ===== 重置小星星的透明度 =====
        让所有小星星变得更透明，设置最大不透明度为0.01
        
        就像给所有小星星洗了个澡，让它们都变得朦胧。
        这通常在训练刚开始时使用，帮助小星星"从头学习"
        """
        # 将所有不透明度限制为最大0.01，然后转换回适合优化的格式
        # 就像告诉所有小星星："先变得几乎看不见，然后慢慢变得更明显"
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        
        # 更新到优化器中
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        """
        # ===== 从文件加载小星星 =====
        从PLY文件中读取保存的小星星信息
        就像打开魔法盒子，让里面保存的小星星再次出现
        
        参数:
            path: PLY文件路径（魔法盒子的位置）
        """
        # 读取PLY文件
        # 就像小心翼翼地打开魔法盒子
        plydata = PlyData.read(path)

        # 读取小星星的位置
        # 就像看看每个小星星应该挂在天空的哪个位置
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
                        
        # 读取小星星的透明度
        # 就像检查每个小星星有多清晰可见
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        # 读取小星星的基础颜色（DC特征）
        # 就像检查每个小星星的主要颜色
        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        # 读取小星星的颜色细节（额外特征）
        # 就像检查每个小星星的花纹和细节
        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # 重塑特征张量
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        # 读取小星星的大小
        # 就像检查每个小星星有多大
        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # 读取小星星的方向
        # 就像检查每个小星星朝向哪里
        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        # 将所有读取的数据转换为PyTorch参数（可学习的形式）
        # 就像把小星星从盒子里拿出来，让它们能再次"活动"
        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        # 设置颜色复杂度为最大值
        # 就像让小星星使用它们所有的颜色能力
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        """
        # ===== 替换优化器中的某个属性 =====
        在优化器中用新的张量替换旧张量，同时保持优化器状态
        就像在不打乱整个书包的情况下，换掉里面的一本书
        
        参数:
            tensor: 新的张量（新书）
            name: 要替换的张量名称（哪本书）
        返回:
            包含更新后的张量的字典
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                # 获取当前存储的优化器状态
                stored_state = self.optimizer.state.get(group['params'][0], None)
                # 重置优化器的动量和方差项
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                # 从优化器状态中删除旧参数
                del self.optimizer.state[group['params'][0]]
                # 创建新参数并设置为需要梯度
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                # 为新参数设置优化器状态
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        """
        # ===== 精简优化器中的点 =====
        根据掩码剪枝优化器中的参数，移除不需要的点
        就像从书包中取出一些不再需要的书，让书包变轻
        
        参数:
            mask: 布尔掩码，表示要保留的点（True表示保留）
        返回:
            包含剪枝后参数的字典
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            # 获取存储的优化器状态
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                # 应用掩码到优化器状态
                # 就像只保留我们想要的书的笔记
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                # 从优化器状态中删除旧参数
                del self.optimizer.state[group['params'][0]]
                # 应用掩码到参数，创建新的参数（只保留想要的部分）
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                # 为新参数设置优化器状态
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                # 如果没有存储状态，只应用掩码到参数
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        """
        # ===== 剪枝小星星 =====
        移除一些不需要的小星星，保持场景整洁
        就像从天空中取下一些不再闪亮的星星
        
        参数:
            mask: 布尔掩码，表示要移除的小星星（True表示移除）
        """
        # 取反掩码，得到要保留的点
        # 把"要移除的"变成"要保留的"
        valid_points_mask = ~mask
        
        # 剪枝优化器中的参数
        # 告诉优化器："这些小星星不用管了"
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        # 更新模型参数为剪枝后的参数
        # 更新我们的小星星集合，只保留好的小星星
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # 更新其他相关数据
        # 也更新小星星的"记录本"
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        """
        # ===== 添加新张量到优化器 =====
        将新的张量添加到优化器中现有的张量
        就像往书包里添加新书，同时保持书包的组织结构
        
        参数:
            tensors_dict: 包含要添加的张量的字典
        返回:
            包含更新后参数的字典
        """
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            # 获取要添加的张量
            extension_tensor = tensors_dict[group["name"]]
            # 获取当前存储的优化器状态
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                # 扩展优化器状态中的动量和方差项
                # 就像为新书准备笔记空间
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                # 从优化器状态中删除旧参数
                del self.optimizer.state[group['params'][0]]
                # 连接旧参数和新参数
                # 就像把新书和旧书放在一起
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                # 为新参数设置优化器状态
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                # 如果没有存储状态，只连接参数
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        """
        # ===== 添加新小星星 =====
        将新创建的小星星添加到现有的模型中
        就像在天空中挂上一些新的小星星
        
        参数:
            new_xyz: 新小星星的位置
            new_features_dc: 新小星星的基础颜色
            new_features_rest: 新小星星的颜色细节
            new_opacities: 新小星星的透明度
            new_scaling: 新小星星的大小
            new_rotation: 新小星星的方向
        """
        # 创建包含所有新参数的字典
        # 就像准备好所有新小星星的属性
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        # 将新参数添加到优化器
        # 告诉优化器："有新的小星星要照顾了"
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        
        # 更新模型参数
        # 更新我们的小星星集合，加入新的小星星
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        # 重置相关计数器
        # 为所有小星星准备新的"记录本"
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        """
        # ===== 分裂小星星 =====
        将大的高斯点分裂成多个较小的高斯点
        就像把一颗大星星分裂成几颗小星星
        
        这就像细胞分裂：一个大细胞分成多个小细胞，
        总体积相同，但能表现更多细节！
        
        参数:
            grads: 梯度（表示哪些点需要更多注意）
            grad_threshold: 梯度阈值，超过此值的点将被分裂
            scene_extent: 场景范围（帮助决定哪些点足够大可以分裂）
            N: 每个点分裂成的新点数量，默认为2（一变二）
        """
        n_init_points = self.get_xyz.shape[0]
        
        # 创建填充梯度张量
        # 就像给每个小星星评分，看哪些需要分裂
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        
        # 选择满足分裂条件的点:
        # 1. 梯度大（说明这个点很重要）
        # 2. 点的大小大于某个阈值（说明这个点足够大可以分裂）
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        # 计算分裂后的参数
        # 就像规划如何分裂星星
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        stds = torch.cat([stds, 0 * torch.ones_like(stds[:,:1])], dim=-1)
        means = torch.zeros_like(stds)
        
        # 使用正态分布采样新点的偏移
        # 就像决定小星星应该放在大星星周围的哪个位置
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        
        # 计算新点的位置
        # 把小星星放在大星星附近的位置
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        
        # 计算新点的缩放（缩小原点的缩放）
        # 小星星比大星星小一些
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        
        # 其他参数保持不变
        # 小星星继承大星星的其他属性
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        # 添加新点到模型
        # 把新的小星星加入到天空中
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        # 创建剪枝掩码，移除原始点，保留新点
        # 移除旧的大星星，只保留新的小星星
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        """
        # ===== 复制小星星 =====
        复制那些特别重要但又很小的星星
        就像看到一颗特别漂亮的小星星，想再来一颗一模一样的！
        
        参数:
            grads: 梯度（表示哪些点特别重要）
            grad_threshold: 梯度阈值，超过此值的点将被复制
            scene_extent: 场景范围
        """
        # 选择那些重要程度高（梯度大）的点
        # 就像找出最闪亮的小星星
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        
        # 但只复制那些足够小的点（与分裂相反，这里是复制小点）
        # 就像只复制小星星，大星星就分裂它们
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        # 复制选定点的所有参数
        # 就像复制小星星的所有属性
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        # 添加克隆的点到模型
        # 把复制的小星星放到天空中
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        """
        # ===== 整理小星星花园 =====
        这个函数做三件事：
        1. 复制重要的小星星
        2. 分裂重要的大星星
        3. 移除不重要的星星
        
        就像园丁照料花园：复制美丽的花，分开拥挤的花丛，
        移除枯萎的花，让整个花园更加美丽！
        
        参数:
            max_grad: 最大梯度阈值（用来决定哪些星星重要）
            min_opacity: 最小不透明度（太透明的星星会被移除）
            extent: 场景范围
            max_screen_size: 最大屏幕尺寸（太大的星星会被分裂）
        """
        # 计算归一化梯度（重要性指标）
        # 就像给每个星星打分
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0  # 处理NaN值

        # 第1步：复制重要的小星星
        self.densify_and_clone(grads, max_grad, extent)
        
        # 第2步：分裂重要的大星星
        self.densify_and_split(grads, max_grad, extent)

        # 第3步：移除不需要的星星
        # 创建剪枝掩码，标记要移除的星星
        # 就像制作一张"要移除的星星"清单
        prune_mask = (self.get_opacity < min_opacity).squeeze()  # 太透明的星星
        
        if max_screen_size:
            # 移除在屏幕上太大的点和在世界中太大的点
            # 就像移除那些太大、太显眼的星星
            big_points_vs = self.max_radii2D > max_screen_size  # 屏幕上太大
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent  # 世界中太大
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
            
        # 执行剪枝操作，移除不需要的星星
        self.prune_points(prune_mask)

        # 清理CUDA缓存（释放内存）
        # 就像打扫花园工具，准备下一次工作
        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        """
        # ===== 记录小星星的重要性 =====
        跟踪每个小星星的重要程度，这些信息将用于决定
        哪些星星需要被复制、分裂或移除。
        
        就像一个小星星日记本，记录每个星星的表现，
        以便我们知道哪些星星需要特别关注！
        
        参数:
            viewspace_point_tensor: 视空间点张量（包含梯度信息）
            update_filter: 更新掩码（表示哪些点需要更新）
        """
        # 累加梯度范数（重要性指标）
        # 就像给星星的表现打分，并记录在它们的成绩单上
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
        
        # 更新计数器（记录更新了多少次）
        # 就像记录我们观察了星星多少次
        self.denom[update_filter] += 1