"""
环形数据CNN二分类
这是一个从原始数据开始训练模型的一个过程：
1. 特征工程
2. 数据预处理
3. 构建网络
4. 模型训练，模型评估，持久化
5. 样本预测


模型欠拟合优化方式：
1. 数据
   样本数量，数据质量，采样方式
2. 模型结构
   模型深度，激活函数选择，池化，归一化，神经元数量
3. 模型参数
   学习率，正则项
4. 训练方式

模型过拟合解决方案：
    正则化：L1（Ridge），L2（Lasso)
    Dropout: 随机删除部分神经元
"""
import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

def load_data():
    # 1. 加载数据
    X, Y = make_circles(
        n_samples=1000,  # 样本数目
        noise=0.1,  # 噪声样本比例
        factor=0.2,  # 内圈直径是外圈直径的factor倍
        random_state=24  # 随机数种子
    )

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=24)
    print(f"训练数据shape形状为: {type(x_train)} - {x_train.shape} -- {type(y_train)} - {y_train.shape}")
    print(f"评估数据shape形状为: {type(x_test)} - {x_test.shape} -- {type(y_test)} - {y_test.shape}")
    print(f"类别取值: {np.unique(y_train)} - {np.bincount(y_train)} -- {np.unique(y_test)} - {np.bincount(y_test)}")

    return x_train, x_test, y_train, y_test


class ClassifyNetwork(nn.Module):
     def __init__(self, in_features: int, num_classes: int):
         """
         分类模型
         :param in_features: 分类模型输入的原始特征向量数目
         :param num_classes: 分类模型对应的类别数目
         """
         super().__init__()
         self.in_features = in_features
         self.num_classes = num_classes
         # 1. 特征提取
         self.features = nn.Sequential(
                 nn.Linear(self.in_features, 8),
                 nn.ReLU(),
                 nn.Linear(8, 8),
                 nn.ReLU()
         )
         # 2. 决策输出
         self.classify = nn.Linear(8, self.num_classes)

     def forward(self, x):
         """
         前向过程
         NOTE:
         bs: 样本数目
         in_features: 每个样本的向量维度大小
         置信度：模型损失计算前的数据对象
         :param x: 输入的原始特征向量，FloatTensor格式，shape形状为: [bs, in_features]
         :return: 前向结果，训练时候一般为置信度值，推理的时候可以直接返回预测结果，FloatTens
        or格式，shape形状为: [bs, num_classes]
         """
         # 1. 样本特征向量提取 [bs,in_features] --> [bs,8]
         features = self.features(x)
         # 2. 基于提取的特征向量进行分类决策 [bs,8] --> [bs,num_classes]
         score = self.classify(features)
         # 3. 基于不同的结果返回不同要求的数据
         if self.training:
             return score
         return torch.softmax(score, dim=1)

def train(x_train, y_train, x_test, y_test):
    # 2. 模型创建
    # 模型结构创建
    net = ClassifyNetwork(in_features=x_train.shape[1], num_classes=len(np.unique(y_train)))
    # 损失函数创建
    loss_fn = nn.CrossEntropyLoss()
    # 优化器创建
    opt = optim.SGD(params=net.parameters(), lr=0.01)

    # 3. 模型训练+模型评估+模型持久化
    total_epoch = 100
    batch_size = 8
    test_batch_size = batch_size * 2
    total_train_batch = len(x_train) // batch_size
    total_test_batch = len(x_test) // test_batch_size + (1 if len(x_test) % test_batch_size != 0 else 0)
    model_output_dir = "./output/02/models"
    os.makedirs(model_output_dir, exist_ok=True)
    for epoch in range(total_epoch):
        # 训练
        net.train()
        train_rnd_indexes = np.random.permutation(len(x_train))
        for batch_idx in range(total_train_batch):
            # 获取当前批次的数据x + y
            si = batch_size * batch_idx
            ei = si + batch_size
            train_batch_indexes = train_rnd_indexes[si: ei]
            batch_x_train = torch.tensor(x_train[train_batch_indexes], dtype=torch.float32)
            batch_y_train = torch.tensor(y_train[train_batch_indexes], dtype=torch. int64)
            # 前向过程
            score = net(batch_x_train)  # [bs,num_classes]
            loss = loss_fn(score, batch_y_train)

            # 反向过程
            opt.zero_grad()  # 重置当前优化器对应的所有参数的梯度为0
            loss.backward()  # 计算和当前损失相同的所有参数的梯度值
            opt.step()  # 参数更新
            print(f"Train Epoch {epoch}/{total_epoch} Batch {batch_idx}/{total_train_batch} Loss:{loss.item():.3f}")

        # 评估
        net.eval()
        with torch.no_grad():
            test_indexes = list(range(len(x_test)))
            for batch_idx in range(total_test_batch):
                # 获取当前批次的数据x + y
                si = test_batch_size * batch_idx
                ei = si + test_batch_size
                test_batch_indexes = test_indexes[si: ei]
                batch_x_test = torch.tensor(x_test[test_batch_indexes], dtype=torch.float32)
                batch_y_test = torch.tensor(y_test[test_batch_indexes], dtype=torch.int64)
                # 前向过程
                score = net(batch_x_test)  # [bs,num_classes]
                loss = loss_fn(score, batch_y_test)
                # 效果评估
                pred_idx = torch.argmax(score, dim=1)  # 获取预测的类别id
                acc = metrics.accuracy_score(batch_y_test.numpy(), pred_idx.numpy())
                print(f"Test Epoch {epoch}/{total_epoch} Batch {batch_idx}/{total_test_batch} "
                      f"Batch-number:{batch_x_test.shape[0]} Loss:{loss.item():.3f} Accuracy: {acc: .3f} ")
        # 模型持久化
        torch.save(
            {
                'net': net,  # 模型对象(参数 + 结构)
                'net_param': net.state_dict(),  # 模型网络对应的所有参数
                'epoch': epoch
            },
            os.path.join(model_output_dir, f"{epoch:06d}.pkl")
        )

def predict(x_test):
    # 1. 模型恢复
    obj = torch.load("./output/02/models/000099.pkl", map_location='cpu', weights_only=False)

    net = obj['net']
    net.eval()  # 进入推理阶段
    print(f"模型恢复完成: \n{net}\n\n")

    # 3. 模型预测
    y_pred_proba = net(x_test)
    print(f"获取预测概率对象为: {type(y_pred_proba)} - {y_pred_proba.shape}")
    return y_pred_proba

if __name__ == '__main__':
    # x_train, x_test, y_train, y_test = load_data()
    # train(x_train, y_train, x_test, y_test)

    # 2. 数据转换
    x = [
        [0.05, -0.01],
        [0.1, 0.3],
        [-0.4, 0.2],
        [1.0, 1.2],
        [0.0, 0.75],
        [0.0, -1.2]
    ]
    x = torch.tensor(x, dtype=torch.float32)
    y_hat = predict(x)
    print(y_hat)
    print(torch.argmax(y_hat, dim=1))