from pathlib import Path
from torchvision import models, datasets, transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from sklearn.model_selection import train_test_split

flower_root_dir = Path(__file__).parent.parent.joinpath("202601_CV/20260412/17flowers")
model_output_path = Path(__file__).parent.parent.joinpath("assets/model")
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available() \
    else torch.device("cpu")

def load_flower_images(class_idx):
    images = []

    if class_idx is None:
        for img_class_path in flower_root_dir.iterdir():
            print("current class path: ", img_class_path)
            for child in img_class_path.iterdir():
                print(child.absolute())
                images.append(str(child.absolute()))
    else:
        class_root = flower_root_dir.joinpath(f"c{class_idx}")
        if not class_root.exists():
            print(f"class {class_idx} not exists")
            return images

        for child in class_root.iterdir():
            print(child.absolute())
            images.append(str(child.absolute()))

    return images
def build_transforms():
    # 数据预处理
    transform_train = transforms.Compose([
        # transforms.Resize(256),  # 缩放
        transforms.Resize(448),  # 缩放
        # transforms.CenterCrop(224),  # 居中裁剪
        transforms.CenterCrop(448),  # 居中裁剪
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(10),  # 随机旋转10度
        transforms.ColorJitter( # 能提升泛化
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.ToTensor(),  # 转为张量
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),  # 缩放
        # transforms.Resize(448),  # 缩放
        transforms.CenterCrop(224),  # 居中裁剪
        # transforms.CenterCrop(448),  # 居中裁剪
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],# RGB均值mean=[0.485,0.456,0.406]
            [0.229, 0.224, 0.225] # RGB标准差std=[0.229,0.224,0.225]
        )
    ])
    return transform_train, transform_val

def load_data():
    """
    划分测试集和训练集，用不同的预处理方式分别处理数据
    :return:
    """
    # 加载数据集
    full_dataset = datasets.ImageFolder(root=flower_root_dir, transform=None)
    # 验证标签分类情况
    # print(full_dataset) # 17类
    # print(full_dataset.classes) # 17类
    print(full_dataset.class_to_idx) # 17类对应的标签
    # print(full_dataset.samples[0]) # 查看一个样本的分类情况
    # print(full_dataset.targets) # 所有样本的标签
    # print(range(len(full_dataset)))

    # 划分训练集和测试集
    train_x, val_x, train_y, test_y = train_test_split(range(len(full_dataset)),
                                                       full_dataset.targets,
                                                       train_size=0.8,
                                                       random_state=42)

    transform_trian, transform_val = build_transforms()
    train_dataset = datasets.ImageFolder(root=flower_root_dir,
                                         transform=transform_trian)
    val_dataset = datasets.ImageFolder(root=flower_root_dir,
                                       transform=transform_val)

    train_dataset.samples = [train_dataset.samples[i] for i in train_x]
    train_dataset.targets = [train_dataset.targets[i] for i in train_x]

    val_dataset.samples = [val_dataset.samples[i] for i in val_x]
    val_dataset.targets = [val_dataset.targets[i] for i in val_x]

    # 数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    return train_loader, val_loader

def load_model() -> models.VGG:
    # 下载模型
    vgg19_bn = models.vgg19_bn(weights=models.VGG19_BN_Weights.DEFAULT)

    # 模型特征层冻结，替换分类层
    for param in vgg19_bn.features.parameters():
        param.requires_grad = False

    # 替换分类层
    # 方案一：只替换最后一层全连接
    vgg19_bn.classifier[6] = nn.Linear(4096, 17)

    # 方案二：可以替换整个分类层
    # vgg19_bn.classifier = nn.Sequential(
    #     nn.Linear(25088, 4096),
    #     nn.ReLU(),
    #     nn.Dropout(0.5),
    #     nn.Linear(4096, 4096),
    #     nn.ReLU(),
    #     nn.Dropout(0.5),
    #     nn.Linear(4096, 17)
    # )
    return vgg19_bn

def fine_tuning():
    # =========================
    # 3. 创建模型
    # =========================
    # 加载模型
    model = load_model().to(device)
    # print(model)
    # for parameter in model.parameters():
    #     print(parameter.requires_grad)

    # 定义损失函数
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 优化器
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=1e-5 # L2 regularization
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                         mode='min',
                                         patience=3,
                                         factor=0.1)

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
        # 切换到训练模式
        model.train()
        # 加载数据
        train_loader, val_loader = load_data()

        train_loss = 0
        train_correct = 0
        train_correct2 = 0
        train_total = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            # 前向传播，计算预测值
            y_hat = model(x)

            # 计算Loss
            loss = loss_fn(y_hat, y)

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
            # 累加预测正确的数量
            probs, indexes = torch.max(y_hat, dim=1)
            for a, b in zip(y, indexes):
                if a == b:
                    train_correct += 1
            # train_correct2 += ((indexes == y)
            #                   .sum()  # True -> 1 预测正确的数量求和
            #                   .item())  # tensor(2) 转成 2

            # 总样本数
            train_total += y.size(0)
            # print(f"epoch: {epoch}, loss: {loss.item()}")

        # 训练集准确率
        train_acc = train_correct / train_total
        train_avg_loss = train_loss / train_total

        # 切换到验证模式
        model.eval()

        # 验证模型
        val_total_loss = 0
        val_total_correct = 0
        val_total = 0
        with torch.no_grad(): # 验证模式不需要计算梯度
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)

                loss = loss_fn(y_hat, y)
                val_loss = loss.item()
                predict = torch.argmax(y_hat, dim=1)
                val_total_correct += (predict == y).sum().item()
                val_total += y.size(0)
                val_total_loss += val_loss

        # 验证集准确率
        val_acc = val_total_correct / val_total
        val_avg_loss = val_total_loss / val_total

        # 更新学习率
        scheduler.step(val_avg_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # ======================================
        # 日志输出
        # ======================================
        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_avg_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Val Loss: {val_avg_loss:.4f} "
            f"Val Acc: {val_acc:.4f} "
            f"LR: {current_lr:.6f}"
        )

        # ======================================
        # Early Stopping
        # ======================================
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            counter = 0

            # 暂存模型
            torch.save(model.state_dict(), model_output_path.joinpath("flower_model.pth"))
            print("✅ Best model saved.")
        else:
            counter += 1
            if counter >= patience:
                print("🛑 Early stopping triggered.")
                break

        # 评价指标
        # 准确率
        # 召回率
        # F值
        # 混淆矩阵
        # ROC 曲线
        # AUC
        # 计算训练集准确率
if __name__ == '__main__':
    # load_flower_images(35)
    # load_model()
    # fine_tuning()
    load_data()