import time
import torch
import torch.nn as nn
import torch.distributed as dist

def main():
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()

    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(512, 1024), nn.ReLU(),
        nn.Linear(1024, 1024), nn.ReLU(),
        nn.Linear(1024, 10),
    )
    ddp = nn.parallel.DistributedDataParallel(model)
    opt = torch.optim.SGD(ddp.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    batch = 64  # per process, so the total batch grows with world size
    x = torch.randn(batch, 512)
    y = torch.randint(0, 10, (batch,))

    def step():
        opt.zero_grad()
        loss = loss_fn(ddp(x), y)
        loss.backward()   # gradients are averaged across processes here
        opt.step()

    for _ in range(5):    # warm-up
        step()

    dist.barrier()
    start = time.time()
    for _ in range(50):
        step()
    dist.barrier()
    step_time = (time.time() - start) / 50

    # Communication cost on its own: all-reduce a buffer the size of the gradients
    n_params = sum(p.numel() for p in model.parameters())
    buf = torch.randn(n_params)
    dist.barrier()
    start = time.time()
    for _ in range(50):
        dist.all_reduce(buf)
    dist.barrier()
    comm_time = (time.time() - start) / 50

    if rank == 0:
        samples_per_sec = world * batch / step_time
        print(f"world_size={world} params={n_params} "
              f"step_ms={step_time*1000:.2f} allreduce_ms={comm_time*1000:.2f} "
              f"samples_per_sec={samples_per_sec:.0f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
