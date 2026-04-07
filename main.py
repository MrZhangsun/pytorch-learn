# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import torch.nn as nn
import torch.nn.functional as f

def embedding_encod():
    num_embeddings = 3
    embedding_dim = 2
    # arr = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
    # inputs = torch.tensor([0, 1, 2])
    # outputs = arr(inputs)
    # print(outputs)

    # input_indices = torch.tensor([0, 1, 2])
    # input_indices: 需要转换编码的类别索引
    # num_classes: 类别数量
    # dummy_encoded = f.one_hot(input_indices, num_classes=num_embeddings)
    # print(dummy_encoded)

    # w = torch.randn(num_embeddings, embedding_dim)
    # print("Embedding权重矩阵（3x2）：")
    # print(w)
    #
    # input_indices = torch.tensor([0, 1, 2])
    # outputs = w[input_indices]
    # print("输入索引对应的编码结果：")
    # print(outputs)

    input_indices = torch.tensor([0, 1, 2])
    dummy_encoded = f.one_hot(input_indices, num_classes=num_embeddings).float()
    print("one-hot编码矩阵：")
    print(dummy_encoded)

    w = torch.randn(num_embeddings, embedding_dim)
    print("embedding权重矩阵：")
    print(w)
    output_equivalent = torch.mm(dummy_encoded, w)
    print("one-hot编码矩阵 x embedding权重矩阵的结果：")
    print(output_equivalent)

    embedding = nn.Embedding(num_embeddings, embedding_dim)
    # 验证等价性
    assert torch.allclose(embedding(input_indices), output_equivalent)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    embedding_encod()
    # print_hi('PyCharm')
    # print(torch.__version__)
    # print(torch.backends.mps.is_available())
    # print(torch.backends.mps.is_built())
    # arr = torch.asarray([1,2,3,4,5,6,7,8,9,10], device='mps')
    # print(arr)
    #
    # device = torch.device("mps")
    # x = torch.rand(5, 3, device=device)
    # y = torch.rand(5, 3, device=device)
    # z = x + y
    # print(z)
    #
    # print(torch.mps.device_count())

    # x = torch.tensor([-100, 0.5,3], device='mps')
    # y = torch.sigmoid(x)
    # print(y)

    # x = torch.tensor([-2.0, 0.0, 2.0])
    # y = torch.tanh(x)
    # print(y)
    # print(torch.sigmoid(x))





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
