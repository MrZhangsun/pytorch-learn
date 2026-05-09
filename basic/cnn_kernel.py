import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 生成1000个合成样本
np.random.seed(42)
X = np.random.randn(1000, 1, 8, 8).astype(np.float32)  # 随机噪声图
y = np.zeros(1000, dtype=np.int64)

# 给后500张图的左上角(0:2, 0:2)区域加上亮块(值设为5.0)
X[500:, :, 0:2, 0:2] += 5.0
y[500:] = 1  # 有亮块的标签为1

# 转为PyTorch张量
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y)


class MiniCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # 1个输入通道 -> 1个输出通道，卷积核大小3x3，步长1，不填充
        self.conv = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU()
        # 经过3x3卷积后，8x8变为6x6，展平后6*6=36维 -> 2分类
        self.fc = nn.Linear(36, 2)

    def forward(self, x):
        x = self.conv(x)          # (1, 6, 6)
        x = self.relu(x)
        x = x.view(x.size(0), -1) # 展平
        x = self.fc(x)
        return x

model = MiniCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=10.0)  # 大学习率加速观察


print("训练前，随机初始化的卷积核：")
print(model.conv.weight.data.squeeze().numpy(), "\n")

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

    # 每轮打印卷积核
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    kernel = model.conv.weight.data.squeeze().numpy()
    print("卷积核:")
    print(np.round(kernel, 2))
    print()


model.eval()
sample_with_block = torch.tensor(X[500:501])   # 有亮块
sample_without = torch.tensor(X[0:1])          # 无亮块

# 单独获取卷积层输出
conv_layer = model.conv
relu = model.relu

out_w = relu(conv_layer(sample_with_block))
out_wo = relu(conv_layer(sample_without))

print("有亮块图片 -> 卷积+ReLU后的特征图（最大值区域对应原图左上角）:")
print(np.round(out_w.detach().numpy().squeeze(), 2))
print("\n无亮块图片 -> 特征图几乎全0:")
print(np.round(out_wo.detach().numpy().squeeze(), 2))