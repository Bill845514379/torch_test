# barrier_effect_demo.py
import torch
import torch.distributed as dist
import os
import time

def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    filename = "shared_log.txt"

    # 每个 rank 模拟不同耗时的前置工作
    time.sleep(2 - rank)  # rank 0 睡 2s, rank 1 睡 1s → rank 1 先完成

    # ❌ 危险：没有 barrier，直接写文件（可能覆盖或交错）
    with open(filename, "a") as f:
        f.write(f"Rank {rank} writing at {time.time():.3f}\n")

    print(f"Rank {rank}: wrote to file.")

    # === 加一个 barrier 再做第二次写入 ===
    dist.barrier()  # 确保所有进程都完成第一次写入

    time.sleep(0.1)  # 小延迟让时间戳更清晰

    with open(filename, "a") as f:
        f.write(f"Rank {rank} writing AFTER BARRIER at {time.time():.3f}\n")

    print(f"Rank {rank}: wrote after barrier.")

    dist.destroy_process_group()

    # 主进程（rank 0）最后打印文件内容
    if rank == 0:
        time.sleep(0.5)  # 等其他进程写完
        print("\n=== File content ===")
        with open(filename, "r") as f:
            print(f.read())
        os.remove(filename)  # 清理


if __name__ == "__main__":
    main()   # ⚠️ 必须通过 main() 调用，不能顶层直接写分布式代码！

# torchrun --nproc_per_node=2 barrier_demo.py
