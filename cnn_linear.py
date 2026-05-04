import torch.nn as nn


linear = nn.Linear(in_features=3, out_features=3, device='cpu', bias=True)
for name, param in linear.named_parameters():
    print()