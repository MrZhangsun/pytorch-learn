import torch
import numpy as np
from humanfriendly.testing import touch

data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
print(x_data)
print(x_data.shape)
print(type(data))
print("*"*60)
nd_array = np.asarray(data)
x_data = torch.from_numpy(nd_array)
print(x_data)
print(x_data.shape)
print(type(data))
print("*"*60)
x_ones = torch.ones_like(x_data)
print(x_ones)
x_rand = torch.rand_like(x_data, dtype=torch.float) # 均匀分布 (Uniform)	[0, 1) 之间的浮点数
print(x_rand)
x_randn = torch.randn_like(x_data, dtype=torch.float) # 标准正态分布 (Normal)	均值为0，方差为1（理论上取值范围 [-∞, +∞]，实际约 [-3, 3] 概率99.7%）
print(x_randn)
print("*"*60)
shape = (2, 3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)
print(rand_tensor)
print(ones_tensor)
print(zeros_tensor)

print("*"*60)
print(torch.cuda.is_available())
print(zeros_tensor.device)
print(zeros_tensor.shape)
print(zeros_tensor.dtype)
print(zeros_tensor.view(1, -1)) # # 重塑形状：1 行，列数自动计算, 原始矩阵的元素数不变
print(zeros_tensor)

print("*"*60)
t_ones = torch.ones((5, 3))
print("First Column: ", t_ones[:, 1])
print("First Row:", t_ones[0])
print("Last Row:", t_ones[-1])
print("Last Column: ", t_ones[:, -1])

t_ones[1, 1] = 0
print(t_ones)
print("*"*60)

t_ones = torch.ones(5, 3)
t_zeros = torch.zeros(5, 3)

"""
函数	        合并方式	        维度变化	            要求
torch.cat	沿现有维度拼接	指定维度的大小增加	    除拼接维度外，其他维度形状必须相同
torch.stack	在新维度上堆叠	总维度数 +1	        所有张量形状必须完全相同
"""
t_cat = torch.cat([t_ones, t_zeros], dim=0)
print(t_cat)
print("*"*60)
t_stack = torch.stack([t_ones, t_zeros], dim=0)
print(t_stack)
print("*"*60)

# This computes the matrix multiplication between two tensors. y1, y2, y3 will have the same value
# ``tensor.T`` returns the transpose of a tensor
data = [[1, 2], [3, 4], [5, 6], [7, 8]]
t_ones = torch.tensor(data)
print(t_ones)
print(t_ones.T)
result1 = t_ones @ t_ones.T # (4, 2) @ (2, 4) = (4, 4)
print(result1)

result2 = torch.matmul(t_ones, t_ones.T)
print(result2)

result3 = None
torch.matmul(t_ones, t_ones.T, out=result3)
print(result3)
print("*"*60)

# mul: 元素相乘， matmul: 矩阵相乘
result4 = torch.mul(t_ones, t_ones)
print(result4)
result5 = torch.mul(t_ones, 2)
print(result5)
t_ones.mul_(3) # in-place操作会在本身元素上进行操作，这种方法都以下划线结尾
print(t_ones)
print("*"*60)

agg_sum = torch.sum(t_ones)
print(agg_sum)
agg_item = agg_sum.item()
print(agg_item, type(agg_item))
print("*"*60)

t_ones = torch.ones(5, 3)
n_ones = t_ones.numpy()
print(type(n_ones)) # <class 'numpy.ndarray'>

t_ones = torch.tensor(n_ones)
print(type(t_ones)) # <class 'torch.Tensor'>

