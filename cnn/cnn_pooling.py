"""
池化：目的是为了降低数据量，让模型关注更明显的信息
最大化池化：取池化窗口中的最大值作为池化结果值
平均池化：取池化窗口中的平均值作为池化结果值
自适应池化：不需要确定池化窗口，滑动步长等池化参数，只需要设定预计要输出的池化结果大小即可，程序会自动计算相关参数。
"""

import torch
import torch.nn as nn
from pathlib import Path
import cv2 as cv

def max_pool_demo(feature_map):
    pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    print("=== 打印池化层配置 ===")
    print(pool)  # 直接打印可以看到所有配置

    print("\n=== 尝试查找可学习参数 ===")
    params = list(pool.named_parameters())
    if not params:
        print("发现：MaxPool2d 没有任何可学习参数 (Weights/Bias)")
    else:
        for name, param in params:
            print(f"参数名称: {name}, 参数形状: {param.shape}")

    print(f"输入到池化模块的数据shape:{feature_map.shape}")
    r = pool(feature_map)
    print(f"池化结果shape:{r.shape}")


def avg_pool_demo(feature_map):
    pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    print("=== 打印池化层配置 ===")
    print(pool)  # 直接打印可以看到所有配置

    print("\n=== 尝试查找可学习参数 ===")
    params = list(pool.named_parameters())
    if not params:
        print("发现：AvgPool2d 没有任何可学习参数 (Weights/Bias)")
    else:
        for name, param in params:
            print(f"参数名称: {name}, 参数形状: {param.shape}")


    print(f"输入到池化模块的数据shape:{feature_map.shape}")
    r = pool(feature_map)
    print(f"池化结果shape:{r.shape}")


def adaptive_pool_demo(feature_map):
    pool = nn.AdaptiveAvgPool2d(output_size=(300, 300))
    print("=== 尝试查找可学习参数 ===")
    params = list(pool.named_parameters())
    if not params:
        print("发现：AdaptiveAvgPool2d 没有任何可学习参数 (Weights/Bias)")
    else:
        for name, param in params:
            print(f"参数名称: {name}, 参数形状: {param.shape}")


    print(f"输入到池化模块的数据shape:{feature_map.shape}")
    r = pool(feature_map)
    print(f"池化结果shape:{r.shape}")



if __name__ == '__main__':
    image = cv.imread(Path(__file__).parent.parent.joinpath("assets/image/xiaoren.png"))
    # 输入数据
    print("输入数据:", image.shape)
    # W x H x C -> C x W x H
    img_tensor = (torch.from_numpy(image)
                  .float()  # 把数据类型从整数（uint8, 0-255）转换成浮点数（float32）。神经网络的计算（卷积、池化）涉及到小数运算和梯度更新，整数类型是无法进行这些数学操作的。
                  .permute(2, 0, 1)  # 维度交换（转置）
                  .unsqueeze(0)  # unsqueeze(0)：在第 0 维增加一个维度。
                  )

    # max_pool_demo(img_tensor)
    # avg_pool_demo(img_tensor)
    adaptive_pool_demo(img_tensor)