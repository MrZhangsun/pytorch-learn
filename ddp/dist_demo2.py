import os
import torch
import torch.distributed as dist

def main():
    rank = int(os.environ["RANK"])

    dist.init_process_group("gloo")

    tensor = torch.tensor([rank], dtype=torch.float32)

    print(f"Before: rank={rank}, tensor={tensor}")

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f"After: rank={rank}, tensor={tensor}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()