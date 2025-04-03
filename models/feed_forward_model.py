import torch
import torch.nn as nn
import torch.nn.functional as F
from scene.gaussian_model import GaussianModel

class GaussianFeatureEncoder(nn.Module):
    """
    编码输入特征（如图像或点云）的神经网络
    
    这个网络将提取输入的特征表示，后续会用于预测高斯点的参数
    """
    def __init__(self, input_channels, feature_dim=256):
        super(GaussianFeatureEncoder, self).__init__()
        
        # 基础卷积特征提取器
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        
        # 全局特征
        self.global_features = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, feature_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # 特征提取
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # 全局特征
        features = self.global_features(x)
        
        return features

class GaussianParameterPredictor(nn.Module):
    """
    预测高斯点参数的网络
    
    这个网络将根据输入特征预测一组高斯点的参数，包括：
    - 位置 (xyz)
    - 特征 (颜色球谐系数)
    - 缩放 (scaling)
    - 旋转 (rotation)
    - 不透明度 (opacity)
    """
    def __init__(self, feature_dim=256, num_gaussians=1000, sh_degree=3):
        super(GaussianParameterPredictor, self).__init__()
        
        self.num_gaussians = num_gaussians
        self.sh_degree = sh_degree
        
        # 计算每个高斯点所需的参数数量
        self.xyz_dim = 3  # 位置
        self.scaling_dim = 2  # 缩放系数
        self.rotation_dim = 4  # 四元数表示的旋转
        self.opacity_dim = 1  # 不透明度
        self.features_dc_dim = 3  # 基础颜色特征
        self.features_rest_dim = 3 * ((sh_degree + 1) ** 2 - 1)  # 额外颜色特征
        
        # 参数预测层 - 使用MLPs预测所有参数
        self.predict_xyz = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_gaussians * self.xyz_dim)
        )
        
        self.predict_scaling = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_gaussians * self.scaling_dim)
        )
        
        self.predict_rotation = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_gaussians * self.rotation_dim)
        )
        
        self.predict_opacity = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_gaussians * self.opacity_dim)
        )
        
        self.predict_features_dc = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_gaussians * self.features_dc_dim)
        )
        
        self.predict_features_rest = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_gaussians * self.features_rest_dim)
        )
        
    def forward(self, features):
        batch_size = features.shape[0]
        
        # 预测各个参数
        xyz = self.predict_xyz(features).view(batch_size, self.num_gaussians, self.xyz_dim)
        scaling = self.predict_scaling(features).view(batch_size, self.num_gaussians, self.scaling_dim)
        rotation = self.predict_rotation(features).view(batch_size, self.num_gaussians, self.rotation_dim)
        opacity = self.predict_opacity(features).view(batch_size, self.num_gaussians, self.opacity_dim)
        features_dc = self.predict_features_dc(features).view(batch_size, self.num_gaussians, 1, self.features_dc_dim)
        features_rest = self.predict_features_rest(features).view(batch_size, self.num_gaussians, 
                                                                 self.features_rest_dim // 3, 3)
        
        # 处理特征形状，使其与GaussianModel兼容
        features_dc = features_dc.transpose(2, 3)  # 调整为[B, N, 3, 1]
        features_rest = features_rest.transpose(2, 3)  # 调整为[B, N, 3, ...]
        
        return {
            "xyz": xyz,
            "scaling": scaling,
            "rotation": rotation,
            "opacity": opacity,
            "features_dc": features_dc,
            "features_rest": features_rest
        }

class FeedForwardGaussianSplatting(nn.Module):
    """
    完整的前馈高斯点模型
    
    这个模型从输入（如图像）预测高斯点参数，然后可以用于渲染新视角
    """
    def __init__(self, input_channels=3, feature_dim=256, num_gaussians=1000, sh_degree=3):
        super(FeedForwardGaussianSplatting, self).__init__()
        
        # 编码器和预测器
        self.encoder = GaussianFeatureEncoder(input_channels, feature_dim)
        self.predictor = GaussianParameterPredictor(feature_dim, num_gaussians, sh_degree)
        
        # 高斯模型参数
        self.sh_degree = sh_degree
        self.num_gaussians = num_gaussians
        
    def forward(self, x):
        # 编码输入
        features = self.encoder(x)
        
        # 预测高斯点参数
        gaussian_params = self.predictor(features)
        
        return gaussian_params
    
    def create_gaussian_model(self, params, batch_idx=0):
        """
        从网络预测的参数创建GaussianModel实例
        
        参数:
            params: 网络预测的参数字典
            batch_idx: 要使用的批次索引
            
        返回:
            gaussian_model: 可用于渲染的GaussianModel实例
        """
        gaussian_model = GaussianModel(self.sh_degree)
        
        # 从预测的参数中提取单个批次的参数
        xyz = params["xyz"][batch_idx].detach().clone()
        scaling = params["scaling"][batch_idx].detach().clone()
        rotation = params["rotation"][batch_idx].detach().clone()
        opacity = params["opacity"][batch_idx].detach().clone()
        features_dc = params["features_dc"][batch_idx].detach().clone()
        features_rest = params["features_rest"][batch_idx].detach().clone()
        
        # 将参数转换为GaussianModel的期望格式
        gaussian_model._xyz = nn.Parameter(xyz)
        gaussian_model._scaling = nn.Parameter(scaling)
        gaussian_model._rotation = nn.Parameter(rotation)
        gaussian_model._opacity = nn.Parameter(opacity)
        gaussian_model._features_dc = nn.Parameter(features_dc)
        gaussian_model._features_rest = nn.Parameter(features_rest)
        
        # 初始化其他必要属性
        gaussian_model.active_sh_degree = self.sh_degree
        gaussian_model.max_sh_degree = self.sh_degree
        gaussian_model.max_radii2D = torch.zeros((gaussian_model.get_xyz.shape[0]), device="cuda")
        
        return gaussian_model 