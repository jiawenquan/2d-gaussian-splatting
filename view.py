import sys
from scene import Scene, GaussianModel
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from gaussian_renderer import render, network_gui
from utils.image_utils import render_net_image
import torch

def view(dataset, pipe, iteration):
    """
    用于查看已训练好的高斯模型的函数
    
    参数:
        dataset: 数据集对象
        pipe: 渲染管道参数
        iteration: 要加载的模型迭代次数
    """
    gaussians = GaussianModel(dataset.sh_degree)  # 初始化高斯模型，使用数据集中的球谐函数度数
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)  # 创建场景，加载指定迭代的模型
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]  # 根据数据集设置背景颜色
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")  # 将背景颜色转换为CUDA张量

    while True:  # 无限循环，等待网络GUI的连接和请求
        with torch.no_grad():  # 禁用梯度计算（因为只是查看，不需要训练）
            if network_gui.conn == None:  # 如果网络连接未建立
                network_gui.try_connect(dataset.render_items)  # 尝试建立连接
            while network_gui.conn != None:  # 当网络连接存在时
                try:
                    net_image_bytes = None  # 初始化图像字节数据
                    # 接收网络GUI的渲染请求
                    custom_cam, do_training, keep_alive, scaling_modifer, render_mode = network_gui.receive()
                    if custom_cam != None:  # 如果请求包含自定义相机视角
                        # 使用高斯模型渲染自定义视角的图像
                        render_pkg = render(custom_cam, gaussians, pipe, background, scaling_modifer)
                        # 将渲染结果转换为网络图像格式
                        net_image = render_net_image(render_pkg, dataset.render_items, render_mode, custom_cam)
                        # 将图像转换为字节格式以便传输
                        net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                    # 创建包含模型信息的指标字典
                    metrics_dict = {
                        "#": gaussians.get_opacity.shape[0]  # 当前高斯点的数量
                        # Add more metrics as needed
                    }
                    # 发送图像数据和指标到网络GUI
                    network_gui.send(net_image_bytes, dataset.source_path, metrics_dict)
                except Exception as e:
                    raise e  # 发生异常时抛出
                    print('Viewer closed')  # 输出查看器关闭信息
                    exit(0)  # 退出程序

if __name__ == "__main__":

    # 设置命令行参数解析器
    parser = ArgumentParser(description="Exporting script parameters")
    lp = ModelParams(parser)  # 添加模型参数
    pp = PipelineParams(parser)  # 添加管道参数
    # 添加网络GUI相关参数
    parser.add_argument('--ip', type=str, default="127.0.0.1")  # GUI服务器IP地址
    parser.add_argument('--port', type=int, default=6009)  # GUI服务器端口
    parser.add_argument('--iteration', type=int, default=30000)  # 要加载的模型迭代次数
    
    # 解析命令行参数
    args = parser.parse_args(sys.argv[1:])
    print("View: " + args.model_path)  # 输出要查看的模型路径
    network_gui.init(args.ip, args.port)  # 初始化网络GUI
    
    # 启动查看器
    view(lp.extract(args), pp.extract(args), args.iteration)

    print("\nViewing complete.")  # 查看完成提示