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
    """
    ResNet（Residual Network）是 2015 年 Google 提出的一种深度学习结构，其设计理念是：
    “残差网络（Residual Network）”
    “残差块（Residual Block）”
    “残差连接（Residual Connection）”
    ResNet 引入了残差块（Residual Block），残差块的组成结构如下图所示：
    残差块的组成结构：
    残差块的输入和输出维度相同，因此，残差块的输出可以直接与输入相加。


    针对一个Block，残差计算过程如下：
    identity = x # 输入特征，也是上一个block的输出特征
    out = conv1(x)
    out = bn1(out)
    out = relu(out)

    out = conv2(out)
    out = bn2(out)

    identity = downsample(x) # downsample 输入的是 x
    out += identity #  F(x)

    当前层是在x的基础上，学习到了残差F(x)，最终也学习到的特征y，即：
            y = F(x) + x

    这里：
        F(x)：当前 block 学到的残差映射
        x：shortcut
        y：当前 block 输出

    真实的多层传播，假设：2 个 residual block。
    第1个 block
    输入：x₁
    输出：y₁ = x₁ + F₁(x₁)

    第2个 block
    注意：
    输入已经变了：x₂ = y₁
    所以，输出：y₂ = x₂ + F₂(x₂)
    代入 x2 = y₁ = x₁ + F₁(x₁)，
    得到：y₂ = x₁ + F₁(x₁) + F₂(x₂)
    第n个block的输入的一般形式：Xⁿ⁺¹ = Xⁿ + Fⁿ(Xⁿ)，可以看出每一层的学习都是在上一层的基础上进行修正

    y_hat = F(xⁿ) + F(xⁿ⁻¹) + F(xⁿ⁻²) ... + F(x¹) + x
    loss = y - y_hat

    因此，网络不是每层重新生成完整特征，而是在已有特征基础上不断进行增量修正。
    反向传播时：
        ∂L/∂x = ∂L/∂y (1 + ∂F(x)/∂x)

    由上述公式可以看出，不管神经网络层数有多深，梯度永远都不会消失

    普通多层卷积后梯度消失的原因是因为，每一层梯度通常：<1，例如：0.9
    多层之间又是相乘的关系，假设有100层，则梯度为：0.9^100 = 0.000026，约等于0，因此梯度会消失

    但是对于残差神经网络，由于每一层：Xⁿ⁺¹ = Xⁿ + Fⁿ(Xⁿ)，
    每一层的梯度：∂Xⁿ⁺¹/∂Xⁿ = ∂Xⁿ/∂Xⁿ + ∂Fⁿ(Xⁿ)/∂Xⁿ
                          = 1 + ∂Fⁿ(Xⁿ)/∂Xⁿ
    这种情况下，理论上来讲，仍然存在梯度消失的可能，比如当：∂Fⁿ(Xⁿ)/∂Xⁿ是一个跟大的负值时，梯度就会趋近于0，但是ResNet解决的是，让梯度
    不再必须完全依赖卷积层。shortcut 提供了一个导数为 1 的恒等梯度通路。因此：
        即使残差分支梯度变小，
        梯度仍可以通过 identity path 稳定传播，
        从而大幅缓解梯度消失问题。

    """
    res = models.resnet34(weights=None)
    print(res)
if __name__ == '__main__':
    # alex_net()
    # vgg_net()
    res_net()
    # google_net()