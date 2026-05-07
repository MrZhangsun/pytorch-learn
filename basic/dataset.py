import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from torchvision.io import decode_image
import os


# 加载torch官方的数据集
dataset_save_path = Path(__file__).parent.parent.joinpath("assets/data")
print(dataset_save_path)
# 训练集数据
train_dataset = datasets.FashionMNIST(
    root=dataset_save_path,
    train=True,
    download=True,
    transform=ToTensor()
)

# 测试集数据
test_dataset = datasets.FashionMNIST(
    root=dataset_save_path,
    train=False,
    download=True,
    transform=ToTensor()
)

print(len(train_dataset))
print(len(test_dataset))
print(type(train_dataset))

# 可视化一下
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

fig = plt.figure(figsize=(10, 10))
fig.suptitle("Samples of Fashion MNIST")

# 随机抽三个数据看看
rows, cols = 3, 3
# for i in range(row * col):
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
    sample, label = train_dataset[sample_idx]
    print(f"sample {i}, index: {sample_idx}, label: {label}")

    fig.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(sample.squeeze(), cmap="gray")

# plt.show()

# 创建自定义数据集Creating a Custom Dataset for your files
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

annotations_file = dataset_save_path.joinpath("annotations_file.csv")
cid = CustomImageDataset(annotations_file, dataset_save_path, transform=ToTensor(), target_transform=lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
print(cid[0])