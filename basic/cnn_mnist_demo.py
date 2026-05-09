from torch.utils.data import DataLoader, random_split
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from pathlib import Path


"""
CNN卷积神经网络过程：
数据准备：
    总数据集：10000
    训练数据集：8000
    验证数据集：2000
    训练批次大小：64
网络结构：
    第一层卷积模型：[in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1]
    第一层卷积输入（NCHW）：[64,1,28,28]
    第一层卷积核大小：[3, 3] 二维卷积核
    第一层激活：ReLU
    第一层池化：滑动窗口大小[2,2]，滑动距离：2像素
    第一层输出：Feature Map [16,14,14]
  
    
    第二层卷积模型：[in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1]
    第二层卷积输入（NCHW）：[64,16,28,28]
    第二层卷积核大小：[16, 3, 3] 三维卷积核，卷积结果进行点积，得到输出Feature Map
    第二层激活：ReLU
    第二层输出：Feature Map [32,28,28]
训练过程：
    一、数据预处理：
        1. 数据集划分，确定测试、验证集，确定批次大小、训练次数；
        2. 选择损失函数，优化器，确定学习率等超参数。
    
    ⚠️按照训练次数，开始训练，在每个批次内部正向传播计算损失、反向传播计算梯度、模型参数验证；
    二、正向传播计算损失：
        1. 按照批次，循环训练集数据，执行forward方法；
            1.1. 随机生成第一层16个卷积核的卷积参数，用于第一层卷积运算；
            1.2. 进行第一层卷积：用16个核的卷积参数，对输入的单通道灰度图进行卷积计算，输出16个卷积结果：Feature Map(28, 28);
            1.3. 激活：对Feature Map中的元素，通过激活函数将小于0的元素转成0，大于等于0的元素保持原值输出，激活的作用是：
                - 削弱弱特征或反向特征的影响，增强卷积核特征，降低计算量
                - 引入非线性，增强模型复杂度，类似人工筛选（非线性判断）特征
            1.4. 池化：输入Feature Map大小[16, 28, 28]，经过滑动窗口大小[2,2]，滑动距离：2像素的最大化池化后输出Feature Map大小：[16, 14, 14]
                - 最大化池化：取滑动窗口中的最大值，作为滑动窗口的输出值
        2. 至此，第一层学习结束，第二层学习重复上述过程；第二层需要注意的区别如下：
            2.1. 第二层卷积核参数也是先随机生成的；
            2.2. 第二层卷积核是三维卷积核[16, 3, 3]，因为第一层的输入是16个Feature Map, 二维卷积核需要卷积16次，所以等价于用三维的卷积核卷积计算一次，
            最终，每个卷积核的卷积结果通过“点积”的方式合并计算输出生成一个大小为：[14，14]的Feature Map。经过激活、池化后，整层输出[32, 7, 7]的Feature Map。
        3. 输出层：
            3.1. 经过第二层学习，最终输出Feature Map大小为：[32, 7, 7]，分类目标是0～9的十个数字，所以输出结果为10个
            3.2. 将32个卷积核与10个输出进行全连接：
                - Feature Map拉平：[N,C,H,W] -> [N,C*H*W]，因为全连接层需要输入向量，因此需要拉平32×7×7=1568
                    [
                      [1,2],     拉平后：[1,2,3,4]
                      [3,4]
                    ]
                    也就是说，将[64,32,7,7]大小的list，按照第一维64，第二维自动计算，即：1568大小，拉成一个二维数组，数组的大小为[64,1568]
                - 随机生成[1568, 10]大小的全连接参数矩阵；10表示输出层有10个分类，1568表示输入层有1568多个输入值，所以需要的模型参数矩阵为[10,1568]
                - 用全连接参数矩阵[1568, 10]与拉平后向量值[64,1568]进行全连接计算，得到[64,10]每个样本关于10个分类的预测值
                - 对于每个样本关于10个分类的预测值，求最大值对应的预测值，即作为：当前样本的分类结果，比如10个样本中，第3个预测值最大，则对应0～9中的2，则
                这个样本的分类结果为：2
            3.3. 通过损失函数，传入标签[64,10]值与预测值[64,10]，计算损失Loss
    三、反向传播计算梯度：
        1. 通过正向计算，得到了每一层的模型参数和最终的损失值Loss；
        2. 梯度下降：根据链式法则，对每一层的模型参数进行求导，得到分摊的损失值，即：梯度 ∇W；
        3. 参数更新：对当前的模型参数，减去梯度值，完成对Loss的分摊，得到最终的模型参数； W:=W−η∇W
        4. 当前批次数据学习完成，按照上述过程 继续后续批次数据的学习，完成本轮（epoch）的学习；
    四、模型校验
        1. 每轮（epoch）学习完成后，需要计算模型的损失，准确率等模型评估指标；
        2. 观察指标的变化情况，设置提前终止条件（Early Stopping），一旦发现模型收敛，应当触发Early Stopping，以提升训练效率。
        
"""
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
# batch_size=64: 表示一次用64个样本计算梯度，属于梯度下降方式BGD/SGD/MBGD中的MBGD，即：小批量梯度下降，目前工业界主流的梯度下降方式
# BGD：每次使用权量数据进行训练，这种方式计算量大，训练速度慢，容易出现梯度爆炸，适合数据量小的场景
# MBGD：每次使用多个样本进行训练，这种方式计算量小，训练速度快，容易出现梯度消失，适合数据量大的场景
# SGD：每次训练随机抽取一个样本进行训练，这种方式训练速度快，但是存在随机性
# ⚠️：梯度下降方式里的SGD表示每次训练随机抽取一个样本进行训练，与优化器中的SGD是两个完全不同的概念，是历史遗留问题。
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
        # x.size(0)：即[64,32,7,7]的0号索引位置的值，即：64
        # -1：表示自动计算，也就是说，将[64,32,7,7]大小的list，按照第一维64，第二维自动计算，即：1568大小，拉成一个二维数组，数组的大小为[64,1568]
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