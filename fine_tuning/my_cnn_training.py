"""
自定义CNN实现分类任务训练过程：
1. 数据集取openml的手写数字；
2. 构建自定义模型，对手写数字数据集进行训练，实现分类目标；
3. 训练过程：
    3.1. 加载数据
    3.2. 数据预处理
    3.3. 构建模型
    3.4. 模型参数设计
    3.5. 模型训练
        i：模型评估
        ii: early stop
    3.6. 模型导出
        i: onnx格式
        ii: jit格式（Just in time静态计算图）
    3.7. 模型加载与推理
    3.8. 模型部署
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms

device = torch.device(torch.accelerator.current_accelerator() if torch.backends.mps.is_available() else "cpu")

def load_data_by_ml(train_size, test_size, batch_size):
    """
    机器学习方式加载openml的mist数据集
    划分测试集和训练集
    openml mnist_784数据集：70000个样本，每个样本784维度，本质是28 × 28 = 784图片拉平后的向量，
    取7000个训练，700个测试
    :return:
    """
    # 加载原始数据
    X, y = fetch_openml(
        "mnist_784",
        version=1,
        return_X_y=True,
        as_frame=False)
    # 将图片恢复成28*28的尺寸，因为拉平后的向量丢失了空间关系，不利于CNN提取特征
    X = np.array(X, dtype=np.float32)
    X = np.reshape(X, (-1, 1, 28, 28))

    # 将标签映射为索引，0-9，用于后续分类后按照索引进行匹配数字
    class_names = np.unique(y)
    label_map = {
        label: idx
        for idx, label in enumerate(class_names)
    }
    y = np.array([label_map[label] for label in y])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=train_size, train_size=test_size, random_state=42)

    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train))
    test_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return train_loader, test_loader, class_names

def load_data_by_dl(train_size, test_size, batch_size):
    """
    通过深度学习方式加载mist数据集
    :return:
    """
    transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    full_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    train_dataset, test_dataset, _ = random_split(
        full_dataset, [train_size, test_size, len(full_dataset) - train_size - test_size],
        generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    class_names = full_dataset.targets.unique()
    return train_loader, test_loader, class_names

class MNISTNet(nn.Module):
    """
    构建自定义模型
    :return:
    """
    def __init__(self):
        super(MNISTNet, self).__init__()

        self.features = nn.Sequential(
            # [64, 1, 28, 28]
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # [64, 32, 28, 28]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [64, 32, 14, 14]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # [64, 64, 14, 14]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [64, 64, 7, 7]
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # [64, 64, 7, 7]
            nn.AdaptiveAvgPool2d((1, 1))
            # [64, 64, 1, 1]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 1 * 1, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        # [64, 1, 28, 28]
        z1 = self.features(x)

        # [64, 64*3*3] -> [64, 10]
        z1 = torch.flatten(z1, 1)
        score = self.classifier(z1)
        return score

def train(model, train_data_loader, test_data_loader):
    # 超参数设置
    epochs = 1000
    # 学习率
    learning_rate = 0.001
    # L2 regularization
    weight_decay = 1e-5
    # 优化器
    optimizer = optim.Adam(model.parameters(),
                           lr=learning_rate,
                           weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5)
    # 损失函数
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    early_stop_patience = 10
    best_loss = float("inf") # 正无穷大
    for epoch in range(epochs):
        # 每一批数据训练
        total_train_loss = 0
        correct_y, predict_y = [], []

        for x, y in train_data_loader:
            x = x.to(device)
            y = y.to(device)
            # 前向传播
            model.train()
            z = model(x)
            loss = criterion(z, y)
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()

            # 统计训练集准确率
            predict = torch.argmax(z, dim=1)
            total_train_loss += loss.item()
            correct_y.extend(y.cpu().numpy())
            predict_y.extend(predict.cpu().numpy())

        # 训练集准确率
        train_acc = metrics.accuracy_score(correct_y, predict_y)

        # 切换到测试模式
        model.eval()
        with torch.no_grad():
            total_test_loss = 0
            correct_y, predict_y = [], []
            for x, y in test_data_loader:
                x = x.to(device)
                y = y.to(device)
                z = model(x)
                loss = criterion(z, y)
                # 统计测试集准确率
                total_test_loss += loss.item()
                predict = torch.argmax(z, dim=1)
                correct_y.extend(y.cpu().numpy())
                predict_y.extend(predict.cpu().numpy())

            # 测试集准确率
            test_acc = metrics.accuracy_score(correct_y, predict_y)
            metrics.classification_report(correct_y, predict_y)
            avg_test_loss = total_test_loss / len(test_data_loader)
            avg_train_loss = total_train_loss / len(train_data_loader)
            scheduler.step(avg_test_loss)

            # Early Stopping
            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                torch.save({
                    "class_names": class_names,
                    "num_features": len(train_data_loader),
                    "net": model,  # 模型对象 + 参数 + 结构  --> 如果直接持久化实例对象，那么在恢复的时候，要求该实例对应的class必须在sys.path环境中找的到
                    "net_param": model.state_dict(),  # 仅持久化模型参数，是一个字典
                    "epoch": epoch
                }, "best_model.pt")
                counter = 0
            else:
                counter += 1
                if counter >= early_stop_patience:
                    print("Early stopping")
                    break

            print(f"Epoch: {epoch + 1}/{epochs}, "
                  f"Train Loss: {avg_train_loss:.4f}, Test/Best Loss: {avg_test_loss:.4f}/{best_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, "
                  f"LR: {optimizer.param_groups[0]['lr']:.4f} "
                  f"Early Stop: {counter}")




if __name__ == '__main__':
    bs = 64

    # the_train_loader, the_test_loader, class_names = load_data_by_ml(bs * 300, bs * 50, bs)
    the_train_loader, the_test_loader, class_names = load_data_by_dl(bs * 600, bs * 100, bs)
    # print(len(the_train_loader.dataset), len(the_test_loader.dataset))
    the_model = MNISTNet()
    print(the_model)

    train(the_model, the_train_loader, the_test_loader)
