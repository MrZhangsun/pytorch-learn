from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from pathlib import Path

# =========================
# 1. 加载数据
# =========================
transform = transforms.ToTensor()
dataset_save_path = Path(__file__).parent.parent.joinpath("assets/data")
# 下载/加载训练数据
full_dataset = datasets.MNIST(root=dataset_save_path,
                               train=True,
                               download=True,
                               transform=transform)
# 划分训练集 / 验证集
train_rate = 0.8
train_size = int(len(full_dataset) * train_rate)
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size],
             generator=torch.Generator().manual_seed(42))
# 创建训练数据集加载器
train_dataloader = DataLoader(train_dataset,
                              batch_size=64,
                              shuffle=True)
val_dataloader = DataLoader(val_dataset,
                              batch_size=64,
                              shuffle=False)

# =========================
# 2. 定义CNN模型
# =========================
class SimpleCNN(nn.Module):
    def __init__(self):
        """
        初始化模型需要的参数计算法
        只是定义，不执行，具体逻辑及模型的设计都在forward方法中定义
        """
        super().__init__()
        # 第一层卷积
        self.conv1 = nn.Conv2d(
            in_channels=1,  # 输入通道数
            out_channels=16,  # 输出通道数
            kernel_size=3,  # 卷积核大小
            stride=1,  # 步长
            padding=1,  # 填充
        )

        # 第二层卷积
        self.conv2 = nn.Conv2d(
            in_channels=16,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        # 池化层
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 激活函数
        self.relu = nn.ReLU()

        # 全连接层
        self.fc = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        """
        这个方法才是真正的模型层定义，定义了模型有多少层，每层的顺序是怎样的。
        :param x: 特征
        :return: 特征
        """
        # 输入参数格式：NCHW
        # N: 批次大小, C: 通道数, H: 高度, W: 宽度
        # [64,1,28,28]
        x = self.conv1(x)

        # [64,16,28,28]
        x = self.relu(x)

        # [64,16,14,14]
        x = self.pool(x)

        # [64,32,14,14]
        x = self.conv2(x)

        # [64,32,14,14]
        x = self.relu(x)

        # [64,32,7,7]
        x = self.pool(x)

        # 拉平：[N,C,H,W] -> [N,C*H*W]，因为全连接层需要输入向量，因此需要拉平32×7×7=1568
        x = x.view(x.size(0), -1)

        # [64,10]
        x = self.fc(x)

        return x

# =========================
# 3. 创建模型
# =========================
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() else 'cpu'

model = SimpleCNN().to(device)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 优化器
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# 学习率调度器
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2
)

# ==========================================
# 4. Early Stopping 参数
# ==========================================

best_val_loss = float("inf")

patience = 5

counter = 0


# =========================
# 5. 开始训练
# =========================

epochs = 30

for epoch in range(epochs):
    # --------------------------------------
    # Train
    # --------------------------------------

    model.train()

    train_loss = 0
    train_correct = 0
    train_total = 0

    """
    先对 Batch 中所有样本Loss求平均，
    再对平均Loss做一次反向传播。
    """
    for images, labels in train_dataloader:
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播，计算预测值，输出为[64, 10]，表示64张图片的预测结果，每个图片的预测结果是一个长度为10的向量，表示10个类别（数字 0~9）的概率
        outputs = model(images)

        # 计算Loss
        # labels是[64,]，表示64张图片的标签，每个标签是一个数字，表示图片的类别（数字 0~9）
        loss = criterion(outputs, labels)

        # 梯度清零：模型梯度每个批次会汇总所有样本的平均梯度，作为本次的梯度，pytorch默认会累所有批次的梯度，由于每个批次的梯度在每轮训练过程
        # 中已经反向传播分配更新到模型参数中了，所以需要将梯度清零
        optimizer.zero_grad()

        # 反向传播，是对一批的图进行统一反向传播，效率更高，梯度会叠加
        loss.backward()

        # 更新参数
        optimizer.step()

        # --------------------------------------
        # 模型验证
        # --------------------------------------
        # 累加loss
        train_loss += loss.item()

        # 计算准确率
        # dim=1，在每一行里找最大值，也就是看哪个数字得分最大
        _, predicted = torch.max(outputs, 1) # 找每张图片“得分最大的类别

        # 当前批次图片的数量
        train_total += labels.size(0)
        # 累加预测正确的数量
        train_correct += ((predicted == labels)
                          .sum() # True -> 1 预测正确的数量求和
                          .item()) # tensor(2) 转成 2
    # 计算准确率
    train_acc = train_correct / train_total

    # --------------------------------------
    # Validation
    # --------------------------------------
    model.eval() # 模型从训练模式切换为评估模式

    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    val_acc = val_correct / val_total
    # 平均loss
    avg_train_loss = train_loss / len(train_dataloader)
    avg_val_loss = val_loss / len(val_dataloader)
    # 更新学习率
    scheduler.step(avg_val_loss)

    current_lr = optimizer.param_groups[0]['lr']
    # ======================================
    # 日志输出
    # ======================================

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Train Loss: {avg_train_loss:.4f} "
        f"Train Acc: {train_acc:.4f} "
        f"Val Loss: {avg_val_loss:.4f} "
        f"Val Acc: {val_acc:.4f} "
        f"LR: {current_lr:.6f}"
    )

    # ======================================
    # Early Stopping
    # ======================================

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        counter = 0

        # 保存最佳模型
        torch.save(model.state_dict(), "best_model.pt")
        print("✅ Best model saved.")
    else:
        counter += 1
        print(f"⚠️ Validation loss not improved: {counter}/{patience}")

    # 停止训练
    if counter >= patience:
        print("🛑 Early stopping triggered.")
        break

print("Training Finished.")