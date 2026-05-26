# -*- coding: utf-8 -*-
"""
Desc : xxx
"""
import os.path
import time

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms, utils, datasets

@torch.no_grad()
def t0():
    # 当给定参数weights的时候，会从网络上下载训练好的模型参数，存储到当前用户根目录下的文件夹: .cache\torch\hub\checkpoints
    # Downloading: "https://download.pytorch.org/models/vgg16-397923af.pth" to C:\Users\Administrator/.cache\torch\hub\checkpoints\vgg16-397923af.pth
    # models.VGG16_Weights.DEFAULT 对应的模型是在ImageNet这个数据集上训练出来的（1000个类别的分类数据）
    # https://blog.csdn.net/winycg/article/details/101722445
    net = models.vgg16(
        weights=models.VGG16_Weights.DEFAULT
    )
    print(net)
    # 从net网络中获取第一个参数中的第一个数值
    print(next(net.parameters()).view(-1)[0])
    # print(next(net.parameters()))
    ts = transforms.Compose([
        transforms.ToTensor()
    ])

    img_path = "../datas/小狗.png"
    img_path = "../datas/飞机.jpg"
    img_path = "../datas/飞机2.jpg"
    img_path = "../datas/小猫.jpg"
    img_path = "../datas/小猫2.jpg"

    # 加载数据
    img = Image.open(img_path)
    img = img.convert("RGB")
    img_tensor = ts(img)
    # 将样本转换为批次
    img_tensor = img_tensor[None]  # [3,h,w] --> [1,3,h,w]
    # print(img_tensor, img_tensor.shape, img_tensor.dtype)

    # 模型预测 [1,3,h,w] --> [1,1000]
    score = net(img_tensor)
    # 求解softmax概率 [1,1000] -> [1,1000]
    probs = torch.softmax(score, dim=1)
    # 求解预测类别id --> 选概率/置信度最大的索引下标 --> [1,1000] --> [1]
    pre_class_id = torch.argmax(score, dim=1)
    print(f"当前图像: {img_path}")
    print(f"预测类别id为: {pre_class_id}")
    print(f"预测类别对应预测概率:{probs[0][pre_class_id[0]]:.4f}")
    print(score.shape)
    # [1,1000] --> ([1,k], [1,k])
    topk_probs, topk_indices = torch.topk(probs, k=5, dim=-1)
    print(f"Top5预测类别id:{topk_indices}")
    print(f"Top5预测类别概率:{topk_probs}")


@torch.no_grad()
def t1():
    net = models.vgg16(
        weights=models.VGG16_Weights.DEFAULT
    )
    ts = transforms.Compose([
        transforms.Resize(160, max_size=320),
        transforms.ToTensor()
    ])
    print(net)

    img_path = "../datas/小狗.png"
    img_path = "../datas/小狗2.png"
    img_path = "../datas/小狗3.png"
    # img_path = "../datas/飞机.jpg"
    # img_path = "../datas/飞机2.jpg"
    # img_path = "../datas/小猫.jpg"
    # img_path = "../datas/小猫2.jpg"
    # img_path = "../datas/c1_image_0016.jpg"

    output_dir = os.path.join("./output", os.path.splitext(os.path.basename(img_path))[0])
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    img = Image.open(img_path)
    img = img.convert("RGB")
    img_tensor = ts(img)
    # 将样本转换为批次
    img_tensor = img_tensor[None]  # [3,h,w] --> [1,3,h,w]
    print(img_tensor, img_tensor.shape, img_tensor.dtype)

    # 模型预测 [1,3,h,w] --> [1,1000]
    score = net(img_tensor)
    # 求解softmax概率 [1,1000] -> [1,1000]
    probs = torch.softmax(score, dim=1)
    # [1,1000] --> ([1,k], [1,k])
    topk_probs, topk_indices = torch.topk(probs, k=5, dim=-1)
    print(f"Top5预测类别id:{topk_indices}")
    print(f"Top5预测类别概率:{topk_probs}")

    # ========================================
    conv1 = net.features[0]
    z1 = conv1(img_tensor)  # [1,64,h,w]
    print(conv1)
    print(z1.shape)
    utils.save_image(
        torch.transpose(z1, dim0=1, dim1=0),  # [1,64,h,w] -> [64,1,h,w]
        os.path.join(output_dir, "layer_1.png"),
        pad_value=0.5  # 在合并图像的时候，图像与图像之间的填充颜色
    )

    # ========================================
    # for i in [2, 4, 5, 10, 31]:
    for i in range(1, 32, 2):
        sub_layers = net.features[:i]
        z = sub_layers(img_tensor)
        utils.save_image(
            torch.transpose(z, dim0=1, dim1=0),  # [1,64,h,w] -> [64,1,h,w]
            os.path.join(output_dir, f"layer_{i}.png"),
            pad_value=0.5  # 在合并图像的时候，图像与图像之间的填充颜色
        )


def build_hook_func(module_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    def hook_func(m, m_args, m_outputs):
        """
        :param m: 当前模块对象
        :param m_args: 当前模块的当前forward的入参，tuple类型
        :param m_outputs: 当前模块的当前forward方法的返回结果，也就是forward的返回值
        :return:
        """
        show_imgs = m_outputs[0:1]  # 仅对当前批次的第一张图像进行可视化 [1,C,H,W]
        show_imgs = torch.transpose(show_imgs, 0, 1)  # [C,1,H,W] 将每个通道看成一个图像
        utils.save_image(
            tensor=show_imgs,
            fp=f"{output_dir}/{module_name}.png",
            pad_value=0.5
        )
        return None

    return hook_func


@torch.no_grad()
def t2():
    # net = models.vgg16(
    #     weights=models.VGG16_Weights.DEFAULT
    # )
    net = models.resnet152(
        weights=models.ResNet152_Weights.DEFAULT
    )
    ts = transforms.Compose([
        transforms.Resize(160, max_size=320),
        transforms.ToTensor()
    ])
    print(net)

    img_path = "../assets/image/小狗.png"
    img_path = "../assets/image/小狗2.png"
    img_path = "../assets/image/小狗3.png"
    # img_path = "../datas/飞机.jpg"
    # img_path = "../datas/飞机2.jpg"
    # img_path = "../datas/小猫.jpg"
    # img_path = "../datas/小猫2.jpg"
    # img_path = "../datas/c1_image_0016.jpg"

    output_dir = os.path.join("./output", os.path.splitext(os.path.basename(img_path))[0], net.__class__.__name__)
    os.makedirs(output_dir, exist_ok=True)

    # 加载数据
    img = Image.open(img_path)
    img = img.convert("RGB")
    img_tensor = ts(img)
    # 将样本转换为批次
    img_tensor = img_tensor[None]  # [3,h,w] --> [1,3,h,w]
    print(img_tensor, img_tensor.shape, img_tensor.dtype)

    # 可视化 通过给模块注册hooks的方式来实现 ========================================
    hook_handles = []
    k = 0
    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d):
            _hook_fn = build_hook_func(
                module_name=f"{k:04d}_{name}_conv",
                output_dir=output_dir
            )
            handle = module.register_forward_hook(hook=_hook_fn)
            hook_handles.append(handle)
            k += 1
        elif isinstance(module, nn.BatchNorm2d):
            _hook_fn = build_hook_func(
                module_name=f"{k:04d}_{name}_bn",
                output_dir=output_dir
            )
            handle = module.register_forward_hook(hook=_hook_fn)
            hook_handles.append(handle)
            k += 1
        elif isinstance(module, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d, nn.AdaptiveMaxPool2d)):
            _hook_fn = build_hook_func(
                module_name=f"{k:04d}_{name}_pool",
                output_dir=output_dir
            )
            handle = module.register_forward_hook(hook=_hook_fn)
            hook_handles.append(handle)
            k += 1

    # 模型预测 [1,3,h,w] --> [1,1000]
    score = net(img_tensor)
    # 求解softmax概率 [1,1000] -> [1,1000]
    probs = torch.softmax(score, dim=1)
    # [1,1000] --> ([1,k], [1,k])
    topk_probs, topk_indices = torch.topk(probs, k=5, dim=-1)
    print(f"Top5预测类别id:{topk_indices}")
    print(f"Top5预测类别概率:{topk_probs}")

    # PS: 当不需要执行hook相关代码的时候，一定要进行删除的操作
    for handle in hook_handles:
        handle.remove()


if __name__ == '__main__':
    t2()
