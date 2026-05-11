"""
在加载的预训练模型基础上进行二次训练（也称为微调，Fine-tuning）是深度学习中非常常见的做法。核心思路是：在别人已训练好的模型基础上，用自己的数据继续训练。
微调 vs 从头训练
方式	        训练速度	        所需数据量	    效果（数据少时）	适用场景
从头训练	    慢	            大量（10万+）	    差	            有海量数据，或任务极其特殊
微调	        快	            较少（几百到几千）	好	            大多数实际应用场景

微调的核心策略：冻结 vs 解冻
冻结（Freeze）：固定某些层的参数，训练时不更新

解冻（Unfreeze）：允许参数在训练中更新
示例1：在 VGG16 上微调 Fashion-MNIST（最常用）
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path

# 1. 准备数据（Fashion-MNIST 是单通道 28x28，需要适配 VGG16）
transform = transforms.Compose([
    transforms.Resize(224),  # VGG16 需要 224x224 输入
    transforms.Grayscale(num_output_channels=3),  # 单通道转3通道
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # ImageNet 标准归一化
])
dataset_save_path = Path(__file__).parent.parent.joinpath("assets/data")
train_dataset = datasets.FashionMNIST(root=dataset_save_path, train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 2. 加载预训练模型
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

# 3. 冻结特征提取层（不需要训练）
for param in model.features.parameters():
    param.requires_grad = False  # 冻结

# 4. 修改分类头（适配 Fashion-MNIST 的10个类别）
in_features = model.classifier[6].in_features  # 原始 VGG16 分类头最后一层的输入维度
model.classifier[6] = nn.Linear(in_features, 10)  # 替换为10分类

# 5. 定义优化器（只训练分类头和未冻结的层）
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)  # 只训练分类头
# 或者：optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# 6. 训练
criterion = nn.CrossEntropyLoss()
device = torch.device(torch.accelerator.current_accelerator() if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.train()

for epoch in range(5):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}")

# 7. 保存微调后的模型
torch.save(model.state_dict(), dataset_save_path.joinpath('finetuned_vgg16_fashionmnist.pth'))