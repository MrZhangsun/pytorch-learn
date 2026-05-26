import os
import torch
import torch.distributed as dist

def main():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(
        backend="gloo"
    )

    print(f"Hello from rank {rank}/{world_size}")

    dist.barrier()

    print(f"Rank {rank} passed barrier")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()