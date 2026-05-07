"""
信用卡欺诈检测

"""

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from credit_card_fraud_dataset import FraudDataset, FraudModel
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
import onnxscript

# 固定随机种子
np.random.seed(42)
# 模拟数据
n_samples = 5000
# 特征向量
X = np.random.randn(n_samples, 10)
# 模拟“欺诈规则”（非线性）
y = (X[:, 0]*2 + X[:, 1]*-1.5 + np.sin(X[:, 2]*3) > 1.5).astype(int)
print(y)
# 加入不平衡（欺诈更少）
mask = np.random.rand(n_samples) > 0.9
print(mask)
y = y * mask
print(y)

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 标准化
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# PyTorch Dataset
train_dataset = FraudDataset(X_train, y_train)
test_dataset = FraudDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=64)

# 模型
model = FraudModel()
criterion = nn.BCELoss()  # 二分类
optimizer = optim.Adam(model.parameters(), lr=0.001) # 优化器, 可以自动寻找最合适的学习率

# 训练
# ✅ 1️⃣ 定义 device（M4 GPU）
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)
# ✅ 2️⃣ 模型搬到 GPU
model = model.to(device)
epochs = 150 # 训练轮数
best_auc = 0
patience = 0
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x_train_batch, y_train_batch in train_dataloader:
        # ✅ 3️⃣ 数据搬到 GPU（关键！！）
        x_train_batch = x_train_batch.to(device)
        y_train_batch = y_train_batch.to(device)

        # 1️⃣ 前向传播
        y_predict = model(x_train_batch)

        # 2️⃣ 计算 loss
        loss = criterion(y_predict, y_train_batch)

        # 3️⃣ 清空梯度（非常重要）
        optimizer.zero_grad()

        # 4️⃣ 反向传播
        loss.backward()

        # 5️⃣ 更新参数
        optimizer.step()

        total_loss += loss.item()

    # ===== 验证阶段 =====
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x_test_batch, y_test_batch in test_dataloader:
            x_test_batch = x_test_batch.to(device)
            y_test_batch = y_test_batch.to(device)

            y_test_predict = model(x_test_batch)

            all_preds.extend(y_test_predict.cpu().numpy())
            all_labels.extend(y_test_batch.cpu().numpy())

    auc = roc_auc_score(all_labels, all_preds)

    # early stop
    if auc > best_auc:
        best_auc = auc
        patience = 0
    else:
        patience += 1

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataloader)}, AUC/best_auc: {auc:.4f}/{best_auc:.4f}")

    if patience > 10:
        print("Early stopping")
        break

model.eval()

dummy_input = torch.randn(1, 10).to(device)
torch.onnx.export(
    model,
    dummy_input,
    "assets/fraud_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=18,
    do_constant_folding=True,
    dynamo=False
)