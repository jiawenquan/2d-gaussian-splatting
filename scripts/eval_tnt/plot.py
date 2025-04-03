# ----------------------------------------------------------------------------
# -                   TanksAndTemples Website Toolbox                        -
# -                    http://www.tanksandtemples.org                        -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2017
# Arno Knapitsch <arno.knapitsch@gmail.com >
# Jaesik Park <syncle@gmail.com>
# Qian-Yi Zhou <Qianyi.Zhou@gmail.com>
# Vladlen Koltun <vkoltun@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ----------------------------------------------------------------------------
#
# This python script is for downloading dataset from www.tanksandtemples.org
# The dataset has a different license, please refer to
# https://tanksandtemples.org/license/

import matplotlib.pyplot as plt
from cycler import cycler


def plot_graph(
    scene,
    fscore,
    dist_threshold,
    edges_source,
    cum_source,
    edges_target,
    cum_target,
    plot_stretch,
    mvs_outpath,
    show_figure=False,
):
    """
    绘制精度和召回率的图表
    
    参数:
    scene: 场景名称
    fscore: F1分数
    dist_threshold: 距离阈值
    edges_source: 源点云直方图边缘
    cum_source: 源点云累积分布（精度）
    edges_target: 目标点云直方图边缘
    cum_target: 目标点云累积分布（召回率）
    plot_stretch: 绘图拉伸因子
    mvs_outpath: 输出路径
    show_figure: 是否显示图形
    """
    f = plt.figure()  # 创建图形
    plt_size = [14, 7]  # 设置图形大小
    pfontsize = "medium"  # 设置字体大小

    ax = plt.subplot(111)  # 创建子图
    label_str = "precision"  # 精度标签
    ax.plot(
        edges_source[1::],
        cum_source * 100,
        c="red",
        label=label_str,
        linewidth=2.0,
    )  # 绘制精度曲线（红色）

    label_str = "recall"  # 召回率标签
    ax.plot(
        edges_target[1::],
        cum_target * 100,
        c="blue",
        label=label_str,
        linewidth=2.0,
    )  # 绘制召回率曲线（蓝色）

    ax.grid(True)  # 添加网格
    plt.rcParams["figure.figsize"] = plt_size  # 设置图形大小
    plt.rc("axes", prop_cycle=cycler("color", ["r", "g", "b", "y"]))  # 设置颜色循环
    plt.title("Precision and Recall: " + scene + ", " + "%02.2f f-score" %
              (fscore * 100))  # 设置标题，包含场景名称和F分数
    plt.axvline(x=dist_threshold, c="black", ls="dashed", linewidth=2.0)  # 添加一条垂直虚线表示阈值位置

    plt.ylabel("# of points (%)", fontsize=15)  # 设置y轴标签
    plt.xlabel("Meters", fontsize=15)  # 设置x轴标签
    plt.axis([0, dist_threshold * plot_stretch, 0, 100])  # 设置坐标轴范围
    ax.legend(shadow=True, fancybox=True, fontsize=pfontsize)  # 添加图例
    # plt.axis([0, dist_threshold*plot_stretch, 0, 100])

    plt.setp(ax.get_legend().get_texts(), fontsize=pfontsize)  # 设置图例文本字体大小

    plt.legend(loc=2, borderaxespad=0.0, fontsize=pfontsize)  # 设置图例位置在左上角
    plt.legend(loc=4)  # 设置图例位置在右下角
    leg = plt.legend(loc="lower right")  # 设置图例位置在右下角

    box = ax.get_position()  # 获取坐标轴位置
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])  # 调整坐标轴位置以留出空间放置图例

    # 将图例放在当前坐标轴的右侧
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.setp(ax.get_legend().get_texts(), fontsize=pfontsize)  # 设置图例文本字体大小
    # 设置输出文件名
    png_name = mvs_outpath + "/PR_{0}_@d_th_0_{1}.png".format(
        scene, "%04d" % (dist_threshold * 10000))  # PNG格式输出文件名
    pdf_name = mvs_outpath + "/PR_{0}_@d_th_0_{1}.pdf".format(
        scene, "%04d" % (dist_threshold * 10000))  # PDF格式输出文件名

    # 保存图形并显示
    f.savefig(png_name, format="png", bbox_inches="tight")  # 保存为PNG格式
    f.savefig(pdf_name, format="pdf", bbox_inches="tight")  # 保存为PDF格式
    if show_figure:
        plt.show()  # 如果需要则显示图形
