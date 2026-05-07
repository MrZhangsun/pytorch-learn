import torch
print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())
print(torch.accelerator.current_accelerator())