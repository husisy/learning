import os
import torch

# export NCCL_IB_HCA=mlx5_0
# python -m torch.distributed.launch --nproc_per_node=2 --master_addr 127.0.0.1 --master_port 23333 demo_launch.py

# do NOT do this with DDP, may cause dead-locak according to documentation
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr 127.0.0.1 --master_port 23333 demo_launch.py
# python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr 127.0.0.1 --master_port 23333 demo_launch.py
if __name__=='__main__':
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    master_addr = os.environ['MASTER_ADDR']
    master_port = os.environ['MASTER_PORT']
    print(f'[rank={rank}] start torch.distributed.init_process_group()')
    torch.distributed.init_process_group(backend='nccl', init_method='env://')
    print(f'[rank={rank}] finish torch.distributed.init_process_group()')
    device = torch.device('cuda')

    torch0 = torch.tensor(rank, device=device)
    torch.distributed.all_reduce(torch0, op=torch.distributed.ReduceOp.SUM)

    ret_ = sum(range(world_size))
    ret0 = torch0.detach().to('cpu').numpy()
    assert abs(ret_ - ret0) < 1e-4

    torch.distributed.destroy_process_group()
