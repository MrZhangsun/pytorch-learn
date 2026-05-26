import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


# =========================
# 1. 构造一个假的数据集
# =========================
class RandomDataset(Dataset):

    def __init__(self, size=1000):
        self.x = torch.randn(size, 10)
        self.y = torch.randn(size, 1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


# =========================
# 2. 简单模型
# =========================
class SimpleModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# 3. 初始化分布式环境
# =========================
def setup():

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    # Mac 使用 gloo
    # PyTorch Distributed 内部会在rank=0的进程中启动一个TCP server，地址：MASTER_ADDR:MASTER_PORT。
    # 其他rank使用TCP client连接MASTER_ADDR:MASTER_PORT，并获取数据。
    dist.init_process_group(
        backend="gloo"
    )

    return rank, world_size, local_rank


# =========================
# 4. 主训练逻辑
# =========================
def main():

    rank, world_size, local_rank = setup()

    print(f"[Rank {rank}] started")

    # -------------------------
    # device
    # -------------------------
    device = torch.device("cpu")
    # if torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # else:
    #     device = torch.device("cpu")

    # -------------------------
    # dataset
    # -------------------------
    dataset = RandomDataset()

    # DistributedSampler:
    # 自动切分数据
    sampler = DistributedSampler(
        dataset,
        shuffle=True
    )

    dataloader = DataLoader(
        dataset,
        batch_size=32,
        sampler=sampler
    )

    # -------------------------
    # model
    # -------------------------
    model = SimpleModel().to(device)

    # DDP 包装
    model = DDP(model)

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3
    )

    # =========================
    # train
    # =========================
    epochs = 3

    for epoch in range(epochs):

        # 每轮重新shuffle
        sampler.set_epoch(epoch)

        epoch_loss = 0.0

        for step, (x, y) in enumerate(dataloader):

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            pred = model(x)

            loss = criterion(pred, y)

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

            if step % 10 == 0:
                print(
                    f"[Rank {rank}] "
                    f"Epoch={epoch} "
                    f"Step={step} "
                    f"Loss={loss.item():.4f}"
                )

        # =========================
        # all_reduce 同步 loss
        # =========================
        loss_tensor = torch.tensor(
            epoch_loss,
            device=device
        )

        dist.all_reduce(
            loss_tensor,
            op=dist.ReduceOp.SUM
        )

        avg_loss = loss_tensor.item() / world_size

        # 只有 rank0 打印
        if rank == 0:
            print(
                f"\n[Epoch {epoch}] "
                f"Global Avg Loss = {avg_loss:.4f}\n"
            )

        # barrier
        dist.barrier()

    # =========================
    # save checkpoint
    # =========================
    if rank == 0:

        torch.save(
            model.module.state_dict(),
            "model.pth"
        )

        print("\nModel saved by rank0")

    # destroy
    dist.destroy_process_group()


if __name__ == "__main__":
    main()