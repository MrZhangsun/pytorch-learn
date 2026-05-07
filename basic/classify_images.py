import torch.accelerator
import torch.nn as nn

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
print(f"Using {device} device")

class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512), # 输入层
            nn.ReLU(),
            nn.Linear(512, 512), # 隐层
            nn.ReLU(),
            nn.Linear(512, 10) # 输出层
        )

    def forward(self, x):
        x = self.flatten(x)
        loss = self.linear_relu_stack(x)
        return loss

if __name__ == '__main__':
    model = NeuralNetwork()
    model.to(device)

    X = torch.rand(size=(2, 28, 28), device=device)
    # logits 含义：模型最后一层（通常是线性层）的直接输出，取值范围 [-∞, +∞]，不是概率
    # 假设模型是10分类问题（如Fashion-MNIST有10个类别），则 logits.shape 为 (1, 10)
    logits = model(X)
    # print(logits)
    pred_probab = nn.Softmax(dim=1)(logits) # 将 logits 转换为概率分布
    print(pred_probab)

    y_pred = pred_probab.argmax(1)
    print(f"Predicted class: {y_pred}")

    print(f"Model structure: {model}\n\n")

    for name, param in model.named_parameters():
        print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")