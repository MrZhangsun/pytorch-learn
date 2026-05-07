"""
逻辑回归 (Logistic Regression) Demo

虽然是"回归"，但实际是二分类模型：
线性层 + Sigmoid → 输出 [0,1] 的概率值
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 生成二分类数据
n_samples = 2000
X_np, y_np = make_classification(
    n_samples=n_samples,
    n_features=10,
    n_informative=8,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42,
)

# 2. 划分 + 标准化
X_train, X_test, y_train, y_test = train_test_split(
    X_np, y_np, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. 转 PyTorch Tensor
X_train = torch.tensor(X_train, dtype=torch.float32, device=device)
y_train = torch.tensor(y_train, dtype=torch.float32, device=device).view(-1, 1)
X_test = torch.tensor(X_test, dtype=torch.float32, device=device)
y_test = torch.tensor(y_test, dtype=torch.float32, device=device).view(-1, 1)


# 4. 逻辑回归模型 = 单层 Linear + Sigmoid
class LogisticRegression(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


model = LogisticRegression(10).to(device)

# BCEWithLogitsLoss 内部做了 sigmoid + BCE 合并计算，数值更稳定
# 这里用 BCELoss 因为 forward 里已经手动加了 sigmoid
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 5. 训练
epochs = 500
for epoch in range(1, epochs + 1):
    model.train()

    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        model.eval()
        with torch.no_grad():
            y_test_pred = model(X_test)
            test_loss = criterion(y_test_pred, y_test)
            acc = ((y_test_pred > 0.5).float() == y_test).float().mean()
            auc = roc_auc_score(y_test.cpu().numpy(), y_test_pred.cpu().numpy())

        print(f"Epoch {epoch:>3d} | "
              f"Train Loss: {loss.item():.4f} | "
              f"Test Loss: {test_loss.item():.4f} | "
              f"Acc: {acc.item():.2%} | "
              f"AUC: {auc:.4f}")

# 6. 查看学习到的权重
print("\n=== 学习到的参数 ===")
for name, param in model.named_parameters():
    print(f"{name}: {param.data.squeeze()}")
