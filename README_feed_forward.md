# 前馈式高斯点渲染系统 (Feed-Forward Gaussian Splatting)

这个项目实现了一个前馈式高斯点渲染系统，可以从单一输入图像生成3D视图，无需通过昂贵的优化过程。

## 基本原理

传统的高斯点渲染系统需要通过长时间的优化来学习表示3D场景的高斯点参数。我们的前馈式方法使用深度神经网络直接从单一输入图像预测高斯点参数，实现快速的新视角合成。

主要组件：
1. **编码器网络**：将输入图像编码为特征表示
2. **参数预测网络**：从特征表示预测高斯点参数
3. **渲染器**：使用预测的高斯点参数渲染新视角

## 系统特点

- **快速推理**：单次前向传播即可渲染新视角，无需优化
- **单视图输入**：只需一张图像即可生成3D表示
- **与传统系统兼容**：使用与原始高斯点渲染相同的渲染器

## 安装指南

确保您已安装以下依赖项：
```bash
pip install torch torchvision numpy pillow matplotlib tqdm
```

## 使用方法

### 1. 训练模型

使用以下命令训练前馈模型：

```bash
python train_feed_forward.py \
    --dataset_path /path/to/your/dataset \
    --image_height 512 \
    --image_width 512 \
    --batch_size 4 \
    --epochs 100 \
    --lr 1e-4 \
    --num_gaussians 2000 \
    --sh_degree 3
```

主要参数说明：
- `dataset_path`：包含训练数据的目录
- `num_gaussians`：每个场景中的高斯点数量
- `sh_degree`：球谐函数的最大阶数（影响颜色表示的复杂度）

### 2. 推理

使用训练好的模型从单一图像渲染新视角：

```bash
python inference_feed_forward.py \
    --model_path /path/to/your/model.pth \
    --image_path /path/to/input/image.jpg \
    --output_dir /path/to/output \
    --n_views 36
```

参数说明：
- `model_path`：训练好的模型路径
- `image_path`：输入图像路径
- `output_dir`：输出目录，用于保存渲染的图像
- `n_views`：要渲染的视图数量

## 数据集格式

训练数据集应具有以下结构：

```
dataset_root/
├── scene1/
│   ├── view1.jpg
│   ├── view2.jpg
│   └── ...
├── scene2/
│   ├── view1.jpg
│   ├── view2.jpg
│   └── ...
└── ...
```

每个场景文件夹包含同一场景的多个视角。在训练时，我们使用第一个视图作为输入，其余视图作为目标。

## 技术细节

1. **高斯点参数**：我们的模型预测以下高斯点参数：
   - 3D位置（xyz）
   - 缩放系数
   - 旋转参数
   - 不透明度
   - 颜色特征（球谐系数）

2. **网络架构**：
   - 编码器：基于卷积神经网络，提取输入图像的特征
   - 参数预测器：基于多层感知机（MLP），预测高斯点参数

3. **损失函数**：
   - L1损失：衡量渲染图像与目标图像的像素差异
   - SSIM损失：衡量渲染图像与目标图像的结构相似性

## 局限性和未来工作

当前实现存在以下局限性：

1. 相机参数处理简化，实际应用中需要精确的相机位姿
2. 性能可能不如传统优化方法精确
3. 仅支持简单场景，复杂场景可能需要更多高斯点

未来工作方向：

1. 改进网络架构，增强特征提取能力
2. 添加深度信息作为额外监督
3. 扩展到视频输入，提高时间连贯性

## 引用

如果您在研究中使用了本项目，请引用原始高斯点渲染论文：

```
@article{kerbl3Dgaussians,
  title={3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  author={Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  journal={ACM Transactions on Graphics},
  year={2023},
  volume={42},
  number={4}
}
``` 