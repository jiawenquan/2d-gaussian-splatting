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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, render_net_image
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    """
    2D高斯散射的主训练函数
    
    参数:
        dataset: 数据集对象，包含训练和测试图像
        opt: 优化参数
        pipe: 渲染管道参数
        testing_iterations: 需要进行测试评估的迭代次数列表
        saving_iterations: 需要保存模型的迭代次数列表
        checkpoint_iterations: 需要保存检查点的迭代次数列表
        checkpoint: 恢复训练的检查点文件路径
    """
    first_iter = 0  # 初始迭代计数器
    tb_writer = prepare_output_and_logger(dataset)  # 准备输出目录和TensorBoard日志记录器
    
    # 步骤2：准备小点点（初始化高斯模型）
    gaussians = GaussianModel(dataset.sh_degree)  # 初始化高斯模型
    scene = Scene(dataset, gaussians)  # 创建场景对象
    gaussians.training_setup(opt)  # 设置高斯模型的训练参数
    
    if checkpoint:  # 如果提供了检查点，从检查点恢复训练
        (model_params, first_iter) = torch.load(checkpoint)  # 加载模型参数和迭代计数
        gaussians.restore(model_params, opt)  # 恢复高斯模型状态

    # 设置背景颜色，根据数据集指定的白色或黑色背景
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")  # 转换为CUDA张量

    # 创建CUDA事件用于计时
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None  # 初始化视点堆栈
    # 初始化指数移动平均损失值（用于日志记录）
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    # 创建进度条
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    # 步骤3-10：开始一遍又一遍地练习画画（主训练循环）
    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()  # 记录迭代开始时间

        gaussians.update_learning_rate(iteration)  # 更新学习率

        # 每1000次迭代增加球谐函数的级别，直到达到最大级别（让点点能表达更丰富的颜色）
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 步骤8：选择一个随机视角（就像从不同角度看物体）
        if not viewpoint_stack:  # 如果视点堆栈为空，重新填充
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))  # 随机选择一个视点
        
        # 步骤3：电脑尝试画出照片（渲染图像）
        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # 步骤4：比较电脑画的和真实照片的不同（计算损失）
        gt_image = viewpoint_cam.original_image.cuda()  # 获取真实图像
        Ll1 = l1_loss(image, gt_image)  # 计算L1损失（颜色差异）
        # 组合L1损失和SSIM损失（SSIM能更好地评估图像质量）
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # 添加更多的训练目标（让表面更平滑，法线更准确）
        lambda_normal = opt.lambda_normal if iteration > 7000 else 0.0  # 7000次迭代后启用法线正则化
        lambda_dist = opt.lambda_dist if iteration > 3000 else 0.0  # 3000次迭代后启用距离正则化

        # 获取渲染的距离和法线信息
        rend_dist = render_pkg["rend_dist"]
        rend_normal = render_pkg['rend_normal'] 
        surf_normal = render_pkg['surf_normal']
        # 计算法线误差（1减去两个法线的点积）
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        # 计算法线损失和距离损失
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # 总损失（所有发现的问题加在一起）
        total_loss = loss + dist_loss + normal_loss
        
        # 反向传播（计算如何调整每个点点）
        total_loss.backward()

        iter_end.record()  # 记录迭代结束时间

        with torch.no_grad():  # 禁用梯度计算的代码块
            # 更新进度条显示的指数移动平均损失
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * dist_loss.item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * normal_loss.item() + 0.6 * ema_normal_for_log

            # 每10次迭代更新进度条
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)  # 更新进度条后缀显示损失值

                progress_bar.update(10)  # 更新进度条
            
            if iteration == opt.iterations:
                progress_bar.close()  # 训练结束时关闭进度条

            # 记录日志和保存模型
            if tb_writer is not None:  # 如果TensorBoard可用
                # 记录距离损失和法线损失到TensorBoard
                tb_writer.add_scalar('train_loss_patches/dist_loss', ema_dist_for_log, iteration)
                tb_writer.add_scalar('train_loss_patches/normal_loss', ema_normal_for_log, iteration)

            # 步骤9：定期检查学习进度（生成训练报告）
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            
            # 在指定的迭代次数保存高斯模型（保存画家的技能）
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # 步骤6-7：添加更多点点或删除无用点点（密集化和剪枝）
            if iteration < opt.densify_until_iter:  # 如果未达到密集化结束迭代次数
                # 更新最大2D半径
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                # 添加密集化统计信息
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                # 在指定迭代次数范围内执行密集化和剪枝操作
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                # 在指定迭代次数或特定条件下重置不透明度
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # 步骤5：调整小点点（优化器更新高斯点的参数）
            if iteration < opt.iterations:
                gaussians.optimizer.step()  # 执行优化器更新
                gaussians.optimizer.zero_grad(set_to_none = True)  # 清零梯度

            # 在指定的迭代次数保存检查点
            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

        # 网络GUI相关代码，处理网络渲染请求
        with torch.no_grad():        
            if network_gui.conn == None:  # 如果网络连接未建立
                network_gui.try_connect(dataset.render_items)  # 尝试连接
            while network_gui.conn != None:  # 当网络连接存在时
                try:
                    net_image_bytes = None
                    # 接收网络GUI的渲染请求
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:  # 如果有自定义相机视角
                        # 渲染自定义视角的图像
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)   
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        # 转换图像为字节格式
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    # 创建指标字典
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0],
                        "loss": ema_loss_for_log
                        # Add more metrics as needed
                    }
                    # 发送数据到网络GUI
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                    if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                        break  # 跳出网络GUI循环，继续训练
                except Exception as e:
                    # raise e
                    network_gui.conn = None  # 发生异常时重置连接

def prepare_output_and_logger(args):    
    """
    准备输出目录和日志记录器
    
    参数:
        args: 包含模型路径的参数对象
        
    返回:
        tb_writer: TensorBoard的SummaryWriter对象或None
    """
    if not args.model_path:  # 如果未指定模型路径
        # 使用作业ID或UUID创建唯一的输出路径
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # 设置输出文件夹
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)  # 创建输出目录
    # 将配置参数写入文件
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # 创建TensorBoard的SummaryWriter
    tb_writer = None
    if TENSORBOARD_FOUND:  # 如果找到TensorBoard
        tb_writer = SummaryWriter(args.model_path)  # 创建SummaryWriter
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

# 步骤10：检查学习成果（测试电脑的绘画能力）
@torch.no_grad()
def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    """
    在特定迭代生成训练报告，评估模型性能
    
    参数:
        tb_writer: TensorBoard的SummaryWriter对象
        iteration: 当前迭代计数
        Ll1: L1损失值
        loss: 总损失值
        l1_loss: L1损失函数
        elapsed: 迭代耗时
        testing_iterations: 需要进行测试的迭代次数列表
        scene: 场景对象
        renderFunc: 渲染函数
        renderArgs: 渲染函数的参数
    """
    if tb_writer:  # 如果TensorBoard可用
        # 记录各类损失和统计信息到TensorBoard
        tb_writer.add_scalar('train_loss_patches/reg_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)

    # 在指定的迭代次数进行测试评估
    if iteration in testing_iterations:
        torch.cuda.empty_cache()  # 清理CUDA缓存
        # 定义验证配置（测试集和训练集样本）
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        # 对每个验证配置进行评估
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0  # 初始化L1测试损失
                psnr_test = 0.0  # 初始化PSNR测试指标
                # 对每个视点进行评估
                for idx, viewpoint in enumerate(config['cameras']):
                    # 渲染当前视点的图像
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    # 如果TensorBoard可用且是前5个视点，记录可视化结果
                    if tb_writer and (idx < 5):
                        from utils.general_utils import colormap
                        # 处理深度图并可视化
                        depth = render_pkg["surf_depth"]
                        norm = depth.max()
                        depth = depth / norm
                        depth = colormap(depth.cpu().numpy()[0], cmap='turbo')
                        # 记录深度图和渲染图到TensorBoard
                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth[None], global_step=iteration)
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)

                        try:
                            # 尝试记录额外的渲染信息（透明度、法线等）
                            rend_alpha = render_pkg['rend_alpha']
                            rend_normal = render_pkg["rend_normal"] * 0.5 + 0.5  # 将法线范围从[-1,1]调整到[0,1]
                            surf_normal = render_pkg["surf_normal"] * 0.5 + 0.5
                            # 记录法线和透明度图到TensorBoard
                            tb_writer.add_images(config['name'] + "_view_{}/rend_normal".format(viewpoint.image_name), rend_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/surf_normal".format(viewpoint.image_name), surf_normal[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/rend_alpha".format(viewpoint.image_name), rend_alpha[None], global_step=iteration)

                            # 处理和记录距离扭曲图
                            rend_dist = render_pkg["rend_dist"]
                            rend_dist = colormap(rend_dist.cpu().numpy()[0])
                            tb_writer.add_images(config['name'] + "_view_{}/rend_dist".format(viewpoint.image_name), rend_dist[None], global_step=iteration)
                        except:
                            pass  # 忽略可能出现的异常

                        # 在第一次测试迭代时记录真实图像（GT）
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    # 累加L1损失和PSNR指标
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                # 计算平均L1损失和PSNR
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                # 打印评估结果
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    # 记录L1损失和PSNR到TensorBoard
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()  # 清理CUDA缓存

if __name__ == "__main__":
    # 准备训练前的设置（就像准备画画前的工具和画布）
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)  # 添加模型参数
    op = OptimizationParams(parser)  # 添加优化参数
    pp = PipelineParams(parser)  # 添加管道参数
    # 添加其他命令行参数
    parser.add_argument('--ip', type=str, default="127.0.0.1")  # GUI服务器IP地址
    parser.add_argument('--port', type=int, default=6009)  # GUI服务器端口
    parser.add_argument('--detect_anomaly', action='store_true', default=False)  # 是否检测异常
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])  # 测试迭代次数
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])  # 保存迭代次数
    parser.add_argument("--quiet", action="store_true")  # 是否安静模式
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])  # 检查点迭代次数
    parser.add_argument("--start_checkpoint", type=str, default = None)  # 起始检查点
    
    # 解析命令行参数
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)  # 将最终迭代次数添加到保存迭代列表
    
    print("Optimizing " + args.model_path)

    # 初始化系统状态（随机数生成器）
    safe_state(args.quiet)

    # 启动GUI服务器，配置并运行训练
    network_gui.init(args.ip, args.port)  # 初始化网络GUI
    torch.autograd.set_detect_anomaly(args.detect_anomaly)  # 设置是否检测异常
    # 开始训练
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # 完成
    print("\nTraining complete.")
