import torch
import torch.distributed as dist
import os

def main():
    print("Program started", flush=True)

    dist.init_process_group(
        backend="gloo",
        init_method="env://"
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    print(f"Rank {rank}/{world_size} initialized", flush=True)

    input_tensor = torch.arange(world_size) + rank * 10
    output_tensor = torch.empty(world_size, dtype=torch.long)

    input_list = list(input_tensor.chunk(world_size))
    output_list = list(output_tensor.chunk(world_size))
    print(input_tensor)
    dist.all_to_all_single(output_tensor, input_tensor)

    print(f"Rank {rank} output: {output_tensor}", flush=True)

    dist.destroy_process_group()

if __name__ == "__main__":
    main()


# torchrun --standalone --nproc_per_node=2 all_to_all_demo.py
