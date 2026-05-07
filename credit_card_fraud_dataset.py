import torch
from torch.utils.data import Dataset
import torch.nn as nn

class FraudDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

class FraudModel(nn.Module):
    # 构建神经网络（核心）
    # 线性 → 激活 → 线性 → 激活 → 输出
    def __init__(self):
        super(FraudModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),  # 输入层 → 隐层1
            nn.ReLU(),
            nn.Linear(32, 16),  # 隐层1 → 隐层2
            nn.ReLU(),
            nn.Linear(16, 1),  # 输出层
            nn.Sigmoid()  # 二分类
        )

    def forward(self, x):
        return self.net(x)
