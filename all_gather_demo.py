# all_gather_demo.py
import torch
import torch.distributed as dist

def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    input_tensor = torch.tensor([rank + 1], dtype=torch.float32)
    output_list = [torch.zeros(1) for _ in range(world_size)]

    print(f"[Rank {rank}] Before all_gather: {input_tensor}")
    dist.all_gather(output_list, input_tensor)
    print(f"[Rank {rank}] After all_gather: {output_list}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

    # torchrun --nproc_per_node=2 all_gather_demo.py
