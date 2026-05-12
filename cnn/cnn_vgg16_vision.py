import cv2 as cv
import torch
from pathlib import Path
import torch.nn as nn
from torchvision import models, transforms, datasets, utils
from PIL import Image

"""
CNN 可视化
"""
dataset_save_path = Path(__file__).parent.parent.joinpath("assets/image")

model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
# print(model) # 打印模型结构

def get_label_by_id(class_idx):
    imagenet_classes = models.VGG16_Weights.DEFAULT.meta['categories']
    # class_idx 是一个 tensor，需要转为 int
    if isinstance(class_idx, torch.Tensor):
        class_idx = class_idx.item()
    return imagenet_classes[class_idx]

# 我们只需要得到结果，不需要更新模型参数，所以不需要记录梯度
@torch.no_grad()
def predict0():

    """
    Compose：意思是“组成”。它接收一个列表（List），列表里包含了一系列的变换操作。
    执行顺序：它会按照你写入列表的先后顺序，依次对图像进行处理。前一步的输出会自动作为后一步的输入。
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)

    # print(next(model.parameters()).shape)
    # print(next(model.parameters()).view(-1).shape)
    # print(next(model.parameters()).view(1, -1).shape)

    # image_path = dataset_save_path.joinpath("小狗.png")
    image_path = dataset_save_path.joinpath("飞机2.jpg")
    img = Image.open(image_path)
    img = img.convert("RGB")
    # print(img.height, ",", img.width)
    # print(img.size)
    img_tensor = transform(img) # 自动会换轴
    # print(img_tensor.shape)
    # img_tensor = torch.unsqueeze(img_tensor, dim=0)
    img_tensor = img_tensor[None] # 在 Python 和 NumPy/PyTorch 中，None 在索引位置等价于 np.newaxis，它的作用是在该位置插入一个新的维度。
    # print(img_tensor.shape)

    # 模型预测 [1,3,h,w] --> [1,1000]
    # [1, 3, 500, 800] --> [1,1000]
    model.eval()
    score = model(img_tensor)
    print(score.shape)
    probs = torch.softmax(score, dim=1)
    print(probs.shape)
    class_id = torch.argmax(probs, dim=1)
    print(class_id)

    print(f"当前图像: {image_path}")
    print(f"预测类别id为: {class_id}")
    print(f"预测类别对应预测概率:{probs[0][class_id[0]]:.4f}")
    print(score.shape)
    print(f"预测类别ID对应预测标签:", get_label_by_id(class_id))

    top_k_prob, top_k_class_id = torch.topk(probs, k=5, dim=1)
    print(f"Top5预测类别:{[get_label_by_id(i.item()) for i in top_k_class_id[0]]}")
    print(f"Top5预测类别概率:{top_k_prob}")


@torch.no_grad()
def predict1():
    transform = transforms.Compose([
        transforms.Resize(160, max_size=320),
        transforms.ToTensor()
    ])

    image_path = dataset_save_path.joinpath("飞机.jpg")
    # image_path = dataset_save_path.joinpath("飞机2.jpg")
    img = Image.open(image_path)
    img = img.convert("RGB")
    img_tensor = transform(img)
    img_tensor = img_tensor[None]
    model.eval()
    score = model(img_tensor)
    probs = torch.softmax(score, dim=1)
    # class_id = torch.argmax(probs, dim=1)
    top_k_prob, top_k_class_id = torch.topk(probs, k=5, dim=1)
    print(f"Top5预测类别id:{top_k_class_id}")
    print(f"Top5预测类别概率:{top_k_prob}")
    print(f"Top5预测类别:{[get_label_by_id(i.item()) for i in top_k_class_id[0]]}")

    conv1 = model.features[0]
    z1 = conv1(img_tensor)
    print(z1.shape)
    print(conv1)

    output_dir = dataset_save_path.joinpath("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir.joinpath("layer_1.png")

    # 保存模型第一层卷积提取的特征图
    utils.save_image(
        torch.transpose(z1, dim0=1, dim1=0),  # [1,64,h,w] -> [64,1,h,w]
        output_file,
        pad_value=0.5  # 在合并图像的时候，图像与图像之间的填充颜色
    )

    print(model)
    """
    可视化 VGG16 模型在不同深度的“特征图”（Feature Maps）。
简单来说，它想展示：随着神经网络层数的加深，AI 眼中的图片是如何从“简单的线条”变成“复杂的图案”的。
    
    这段代码是深度学习中的**“开天眼”**操作。它打破了神经网络的“黑盒”，
    让你亲眼看到模型是如何一步步把一张彩色照片拆解成数学特征的。
    这对于理解 CNN（卷积神经网络）的工作原理非常有帮助！
    
    CNN 多层卷积的过程，类似学习物体指纹信息的过程
    """
    for i in [2, 4, 5, 10, 31]: # 这是在选取 VGG16 模型中几个具有代表性的层索引。
        sub_layers = model.features[:i]
        z = sub_layers(img_tensor) # 前向传播：把图片喂给这个“半成品”模型, 得到这个“半成品”模型所提取的特征图。
        output_file = output_dir.joinpath(f"layer_{i}.png")
        utils.save_image(
            torch.transpose(z, dim0=1, dim1=0),
            output_file,
            pad_value=0.5
        )

if __name__ == '__main__':
    predict1()