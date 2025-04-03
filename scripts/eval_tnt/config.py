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

# 全局参数定义 - 请勿修改
# 这个字典定义了不同场景的距离阈值tau (单位: 米)
# 距离阈值tau是评估时使用的关键参数，决定了点云间的匹配距离阈值
# 每个场景根据其尺寸特点有不同的适合阈值：
# - 较大的场景(如Church、Courthouse)使用较大的tau值(0.025)
# - 中等场景(如Barn、Meetingroom)使用中等tau值(0.01)
# - 较小或细节丰富的场景(如Caterpillar、Ignatius、Truck)使用较小的tau值(0.003-0.005)
scenes_tau_dict = {
    "Barn": 0.01,        # 谷仓场景，中等尺寸，tau=0.01
    "Caterpillar": 0.005, # 毛毛虫场景，细节丰富，tau=0.005
    "Church": 0.025,     # 教堂场景，大型建筑，tau=0.025
    "Courthouse": 0.025, # 法院场景，大型建筑，tau=0.025
    "Ignatius": 0.003,   # Ignatius雕像，细节非常丰富，tau=0.003
    "Meetingroom": 0.01, # 会议室场景，中等大小，tau=0.01
    "Truck": 0.005,      # 卡车场景，细节丰富，tau=0.005
}
