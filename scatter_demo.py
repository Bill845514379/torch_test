# scatter_demo.py
import torch
import torch.distributed as dist

def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if rank == 0:
        scatter_list = [torch.tensor([i * 10 + 1], dtype=torch.float32) for i in range(world_size)]
        recv_tensor = torch.zeros(1)
        print(f"[Rank {rank}] Sending scatter data: {scatter_list}")
        dist.scatter(recv_tensor, scatter_list, src=0)
    else:
        recv_tensor = torch.zeros(1)
        dist.scatter(recv_tensor, None, src=0)

    print(f"[Rank {rank}] Received: {recv_tensor}")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()


    # torchrun --nproc_per_node=2 scatter_demo.py
