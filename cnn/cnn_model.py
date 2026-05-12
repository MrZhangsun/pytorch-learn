import cv2 as cv
import torch
from pathlib import Path
import torch.nn as nn


def format_image(image_path):
    image = cv.imread(image_path)

    image = cv.resize(image, (100, 200))

    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    image = image / 255.0 - 0.5 # 去中心化

    image = torch.tensor(image, dtype=torch.float32)

    image = torch.permute(image, (2, 0, 1))

    return image

def cnn0(image):
    # 输入一张3通道图，输出6个神经元
    conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1)
    act1 = nn.ReLU()
    pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    conv2 = nn.Conv2d(in_channels=6, out_channels=10, kernel_size=3, stride=1, padding=1)
    act2 = nn.Sigmoid()
    pool21 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
    pool22 = nn.AdaptiveAvgPool2d(output_size=(8, 8))

    fc1 = nn.Linear(in_features=10*8*8, out_features=256)
    act3 = nn.ReLU()
    fc2 = nn.Linear(in_features=256, out_features=2)

    # 加载图像，并且将图像转换为Tensor对象 [bs,3,h,w]
    image = torch.unsqueeze(image_tensor, 0)
    print("第一层输入：", image.shape)
    z1 = conv1(image)
    print("第一层卷积输出：", z1.shape)
    z2 = act1(z1)
    print("第一层激活输出：", z2.shape)
    z3 = pool1(z2)
    print("第一层池化输出：", z3.shape)
    z4 = conv2(z3)
    print("第二层卷积输出：", z4.shape)
    z5 = act2(z4)
    print("第二层激活输出：", z5.shape)
    z6 = pool21(z5)
    print("第二层池化输出：", z6.shape)
    z7 = pool22(z5)
    print("第二层自适应池化输出：", z7.shape)
    z8 = torch.flatten(z7, start_dim=1)
    # z8 = z7.reshape(-1, 10 * 8* 8)
    print("展平输出：", z8.shape)
    z9 = fc1(z8)
    print("全连接层1输出：", z9.shape)
    z10 = act3(z9)
    print("全连接层2输出：", z10.shape)
    z11 = fc2(z10)
    print("全连接层3输出：", z11.shape)
    print("最终输出：", z11)

def cnn1(image):
    features = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        nn.Conv2d(in_channels=6, out_channels=10, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        nn.AdaptiveAvgPool2d(output_size=(8, 8)),
    )

    classify = nn.Sequential(
        nn.Linear(in_features=10*8*8, out_features=256),
        nn.ReLU(),
        nn.Linear(in_features=256, out_features=2),
    )

    print("特征层：", features)
    print("分类层：", classify)

    image = torch.unsqueeze(image, 0)
    print("输入：", image.shape)
    z7 = features(image)
    print("特征层输出：", z7.shape)
    z11 = classify(z7.flatten(start_dim=1))
    print("分类层输出：", z11.shape)

if __name__ == '__main__':
    image_path2 = Path(__file__).parent.parent.joinpath("assets/image/xiaoren.png")
    image_tensor = format_image(image_path2)

    cnn1(image_tensor)


