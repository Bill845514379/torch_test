import torch
import torch.distributed as dist
import os

def main():
    dist.init_process_group(
        backend="gloo",  # CPU 用 gloo
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    tensor = torch.ones(1) * (rank + 1)

    print(f"Before all_reduce Rank {rank}: {tensor}")

    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    print(f"After all_reduce Rank {rank}: {tensor}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

# 启动方式：
# torchrun --nproc_per_node=2 all_reduce_demo.py
