from torchvision import models


def alex_net():
    """
    重大突破：
        使用 GPU 训练
        使用 ReLU
        使用 Dropout
        ImageNet 大幅领先传统方法

        AlexNet 的特点：
        8 层
        大卷积核（11×11、5×5）
        参数量巨大
        结构还比较粗糙
    :return:
    """
    alex = models.alexnet(weights=None)
    # alex = models.alexnet(weights=models.AlexNet_Weights.DEFAULT)
    print(alex)

def zf_net():
    pass

def vgg_net():
    """
    VGG（Visual Geometry Group Network）是由 Visual Geometry Group 在 2014 年提出的一类卷积神经网络（CNN）结构，论文作者主要是 Karen Simonyan 和 Andrew Zisserman。
    它最经典的版本是：
    VGG16
    VGG19
    名字中的数字表示“带参数层”的数量（卷积层 + 全连接层）。
    VGG 在 2014 年的 ImageNet Large Scale Visual Recognition Challenge 2014 中取得了非常优秀的成绩，成为 CNN 历史上的里程碑模型之一。

    VGG 的设计理念非常“暴力但优雅”：“不断堆叠小卷积核，让网络更深。”
    VGG 最大创新：使用多个 3×3 小卷积替代大卷积
    例如：
    以前：一个 7×7 卷积
    VGG：三个 3×3 卷积连续堆叠
    因为：3 个 3×3 卷积的感受野 ≈ 7×7
    但：
        参数更少
        非线性更多
        表达能力更强
    :return:

    """
    vgg = models.vgg16(weights = None)
    print(vgg)

def google_net():
    google = models.googlenet(weights=None)
    print(google)

def res_net():
    res = models.resnet34(weights=None)
    print(res)
if __name__ == '__main__':
    # alex_net()
    # vgg_net()
    # res_net()
    google_net()