import os
import glob
import json
import pandas as pd
import argparse

def main(result_dirs):
    # 查找所有结果JSON文件
    results = glob.glob(os.path.join(result_dirs, '*', 'results.json'))

    # 初始化列表来存储数据
    data = []

    # 读取每个JSON文件并提取指标
    for result_file in results:
        with open(result_file, 'r') as f:
            result = json.load(f)
        
        # 从文件路径中提取实验名称
        exp_name = result_file.split('/')[-2]
        
        # 提取指标：PSNR(峰值信噪比)、SSIM(结构相似性)和LPIPS(感知相似性)
        psnr = result['ours_30000'].get('PSNR', 'N/A')
        ssim = result['ours_30000'].get('SSIM', 'N/A')
        lpips = result['ours_30000'].get('LPIPS', 'N/A')
        
        # 将数据添加到列表中
        data.append({
            'Experiment': exp_name,
            'PSNR': psnr,
            'SSIM': ssim,
            'LPIPS': lpips
        })

    # 从收集的数据创建DataFrame
    df = pd.DataFrame(data)

    # 按实验名称排序DataFrame
    df = df.sort_values('Experiment')

    # 显示表格
    print(df.to_string(index=False))

    # 计算平均PSNR, SSIM和LPIPS值
    avg_psnr = df['PSNR'].mean()
    avg_ssim = df['SSIM'].mean()
    avg_lpips = df['LPIPS'].mean()

    # 打印平均指标
    print(f"Average PSNR: {avg_psnr}")
    print(f"Average SSIM: {avg_ssim}")
    print(f"Average LPIPS: {avg_lpips}")

    # 可选：将表格保存为CSV文件
    # df.to_csv('results_summary.csv', index=False)

if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="Process results from JSON files.")
    parser.add_argument("--model_path", "-m", help="model path")
    args = parser.parse_args()
    # 调用主函数处理结果
    main(args.model_path)