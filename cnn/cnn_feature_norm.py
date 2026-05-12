"""
CNN特征归一化

Norm 的本质：
    对网络中间特征做标准化，
    让数据分布更稳定，
    从而让深层网络更容易训练。
Norm 的核心思想，就是：
    强行把数据：
    拉回稳定分布。
公式： x^ = (x−μ) / sqrt((σ² + ϵ))
    x：当前特征值
    μ：特征值的均值
    σ：特征值的方差
    ϵ：一个极小值，避免除0错误
    x^：得到标准正态分布附近的数据。
缩放、平移：y=γx^+β，归一化后的数据分布被强行固定，可能损失模型表达能力。因此需要根据误差进行缩放和平移，这个缩放和平移程度
    通过模型学习得到，γ和β本质也是模型参数，通过正向传播计算损失，反向传播计算梯度，梯度下降更新参数的方式计算得到。

Norm 阶段：一般在卷积之后，激活函数之前，输入是feature map

CNN 最经典的 Norm有：
    BatchNorm（BN）：全称Batch Normalization，这是 CNN 时代的革命性技术
    LayerNorm（LN）：Transformer 最核心。
    InstanceNorm（IN）: 按照通道计算
    GroupNorm（GN）：多个InstanceNorm一起计算，Google提出的

所有Norm的计算过程都是一样的，如下：
    1. 计算方差和均值；
    2. 卷积后，输入feature map, 均值，方差，代入公式：
        x^ = (x−μ) / sqrt((σ² + ϵ))
        求解得到归一化后的feature map;
    3. 对feature map进行平移，缩放。
    4. 后续激活函数操作。

真正的区别，只在：μ 和 σ² 到底怎么统计，假设输入是[64,32,28,28]
    BatchNorm（BN）：
        计算方式：计算每个通道“当前batch”的所有特征图的均值和方差，举例：输入[N, C, H, W]，对于单个通道C(卷积核/神经元)，输入数据
        为：[N, H, W]，则，输入总特征数量：N * H * W，对于每个通道来讲，是跨batch进行计算的：
            μ = sum(x) / N * H * W
            σ = sum((x - μ)²) / N * H * W

    InstanceNorm（IN）:
        计算方式：按照每个通道中，每张图的每个通道（卷积核/神经元）进行单独计算均值和方差，举例，输入：[N, C, H, W]
            μ = sum(x) / H * W
            σ = sum((x - μ)²) / H * W

    GroupNorm（GN）：
        计算方式，选择C个通道（这里的C不是全部通道，是选择的几个通道），计算：
            μ = sum(x) / C * H * W
            σ = sum((x - μ)²) / C * H * W

    LayerNorm（LN）：Transformer 最核心。
        计算方式：按照样本维度计算均值和方差。比如样本输入了一张RGB图，通道：C，高度：H，宽度：W，则：
            μ = sum(x) / C * H * W
            σ = sum((x - μ)²) / C * H * W
"""
from pathlib import Path

import torch
import torch.nn as nn
import cv2 as cv
from torchvision import models

def fc_bn():
    fc1 = nn.Linear(in_features=10, out_features=5)
    bn1 = nn.BatchNorm1d(num_features=5, momentum=0.1)

    for name, tensor in bn1.state_dict().items():
        print(f"参数 {name} {tensor.shape} {tensor.dtype} {tensor.device} {tensor.requires_grad}")
    print("=" * 50)

    # 上一个模块的输出特征向量，表示2个样本，每个样本有5个特征
    x = torch.rand(2, 10)
    print(x.shape)
    z1 = fc1(x)
    print(z1.shape)
    z2 = bn1(z1)
    print(z2.shape)

    print("=" * 50)
    print("===== BN输入 =====")
    print("mean:", torch.mean(z1, dim=0))
    print("var :", torch.var(z1, dim=0, unbiased=False))

    print()

    print("===== BN输出 =====")
    print("mean:", torch.mean(z2, dim=0))
    print("var :", torch.var(z2, dim=0, unbiased=False))

    print()

    print("===== running statistics =====")
    print("running_mean:", bn1.running_mean)
    print("running_var :", bn1.running_var)

def conv_bn():
    bn = nn.BatchNorm2d(num_features=6, momentum=0.1)
    conv = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)

    for name, tensor in bn.state_dict().items():
        print(f"参数 {name} {tensor.shape} {tensor.dtype} {tensor.device} {tensor.requires_grad}")

    print("=" * 50)
    image_path = Path(__file__).parent.parent.joinpath("assets/image/xiaoren.png")
    image = cv.cvtColor(cv.imread(image_path), cv.COLOR_BGR2RGB)
    image = cv.resize(image, (10, 10))
    image = torch.tensor(image, dtype=torch.float32)
    image = torch.permute(image, (2, 0, 1))
    image = torch.unsqueeze(image, 0)
    print(image.shape)
    z1 = conv(image)
    print(z1.shape)
    z2 = bn(z1)
    print(z2.shape)

    print("=" * 50)
    print("===== BN输入 =====")
    print("mean:", torch.mean(z1, dim=0))
    print("var :", torch.var(z1, dim=0, unbiased=False))

    print()

    print("===== BN输出 =====")
    print("mean:", torch.mean(z2, dim=0))
    print("var :", torch.var(z2, dim=0, unbiased=False))

    print()

    print("===== running statistics =====")
    print("running_mean:", bn.running_mean)
    print("running_var :", bn.running_var)


def layer_norm():
    ln = nn.LayerNorm(
        normalized_shape=[6, 3, 3]
    )

    print("=" * 50)
    for name, param in ln.named_parameters():
        print(f"参数 {name} {param.shape} {param.dtype} {param.device} {param.requires_grad}")
    print("=" * 50)
    for name, tensor in ln.state_dict().items():
        print(f"参数 {name} {tensor.shape} {tensor.dtype} {tensor.device} {tensor.requires_grad}")
    print("=" * 50)

    # 上一个模块的输出特征向量，表示2个样本，每个样本有6个FeatureMap, 每个Feature Map是3*3的大小
    x = torch.rand(2, 6, 3, 3)
    r = ln(x)
    print(x.shape)
    print(r.shape)


def t4():
    net = models.vgg16_bn(
        weights=models.VGG16_BN_Weights.DEFAULT
    )
    print(net)

    bn1: nn.BatchNorm2d = net.features[1]
    print("缩放系数 ", bn1.weight)
    print("平移系数 ", bn1.bias)
    print("累计的均值 ", bn1.running_mean)
    print("累计的方差 ", bn1.running_var)

if __name__ == '__main__':
    # fc_bn()
    # conv_bn()
    # layer_norm()
    t4()