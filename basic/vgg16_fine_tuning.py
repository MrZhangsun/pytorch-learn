
"""
预训练过程：
1. 加载预训练模型；
2. 从ImageNet下载训练数据集，并划分测试集和验证集；
3. 冻结卷积层，只训练分类器；
4. 进行预训练，观察模型指标，early stop
"""

import torch
from torchvision import datasets, models
from torch.utils.data import Dataset,DataLoader
import os


def load_vgg16_model(use_pretrained=True):
    """
        加载 VGG16 模型
        :param use_pretrained: 是否使用预训练权重
        :return: model: VGG16 模型
    """
    if use_pretrained:
        # 使用预训练权重, DEFAULT = IMAGENET1K_V1
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    else:
        model = models.vgg16(weights=None)

def load_data():
    train_dataset = datasets.ImageFolder(root='', transform=None)
    pass

def train_model():
    pass

if __name__ == '__main__':
    pass