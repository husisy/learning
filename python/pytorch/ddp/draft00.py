import os
import torch
import numpy as np

hfe = lambda x,y,eps=1e-3: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

def demo_send_recv(rank, world_size):
    print(f'# demo_send_recv[rank={rank}]')
    assert world_size==2
    # nccl not support send/recv
    torch.distributed.init_process_group(backend='gloo',
            init_method='tcp://127.0.0.1:23333', rank=rank, world_size=world_size)

    torch0 = torch.empty(3, dtype=torch.float32, device=torch.device('cpu'))
    if rank==0:
        torch.randn(*torch0.shape, out=torch0)
        print(f'[rank={rank}] send "{torch0}"')
        torch.distributed.send(torch0, dst=1)
    else:
        torch.distributed.recv(torch0, src=0)
        print(f'[rank={rank}] recv "{torch0}"')

    torch.distributed.destroy_process_group()


def demo_non_blocking_send_recv(rank, world_size):
    print(f'# demo_non_blocking_send_recv[rank={rank}]')
    assert world_size==2
    # nccl not support isend/irecv
    torch.distributed.init_process_group(backend='gloo',
            init_method='tcp://127.0.0.1:23333', rank=rank, world_size=world_size)

    torch0 = torch.empty(3, dtype=torch.float32, device=torch.device('cpu'))
    if rank==0:
        torch.randn(*torch0.shape, out=torch0)
        print(f'[rank={rank}] isend "{torch0}"')
        request0 = torch.distributed.isend(torch0, dst=1)
    else:
        request0 = torch.distributed.irecv(torch0, src=0)
        print(f'[rank={rank}] irecv "{torch0}"') #some dummy print
    request0.wait()
    print(f'[rank={rank}] torch0={torch0}')

    torch.distributed.destroy_process_group()


def demo_tensor_sync_gradient(rank, world_size):
    print(f'# demo_tensor_sync_gradient[rank={rank}]')
    assert world_size==2
    # nccl not support send/recv
    torch.distributed.init_process_group(backend='gloo',
            init_method='tcp://127.0.0.1:23333', rank=rank, world_size=world_size)

    torch0 = torch.empty(3, dtype=torch.float32, device=torch.device('cpu'), requires_grad=True)
    if rank==0:
        torch.randn(*torch0.shape, out=torch0.data)
        torch0.grad = torch.randn_like(torch0)
        print(f'[rank={rank}] send "{torch0}"')
        print(f'[rank={rank}] torch0.grad= "{torch0.grad}"')
        torch.distributed.send(torch0, dst=1)
    else:
        torch.distributed.recv(torch0, src=0)
        print(f'[rank={rank}] recv "{torch0}"')
        print(f'[rank={rank}] torch0.grad= "{torch0.grad}"')
    print(f'[rank={rank}] totally not work at all') #https://pytorch.org/docs/stable/notes/multiprocessing.html

    torch.distributed.destroy_process_group()


def demo_all_reduce_basic(rank, world_size):
    print(f'# demo_all_reduce_basic[rank={rank}]')
    assert world_size>2
    torch.distributed.init_process_group(backend='nccl',
            init_method='tcp://127.0.0.1:23333', rank=rank, world_size=world_size)
    device = torch.device(f'cuda:{rank}')

    torch0 = torch.tensor(rank, device=device)
    torch.distributed.all_reduce(torch0, op=torch.distributed.ReduceOp.SUM) #default value for group is okay

    ret_ = sum(range(world_size))
    ret0 = torch0.detach().to('cpu').numpy()
    assert hfe(ret_, ret0) < 1e-4

    torch.distributed.destroy_process_group()


def demo_all_reduce(rank, np_state_list):
    print(f'# demo_all_reduce[rank={rank}]')
    world_size = len(np_state_list)
    assert world_size>2
    torch.distributed.init_process_group(backend='nccl',
            init_method='tcp://127.0.0.1:23333', rank=rank, world_size=world_size)
    np_state = np.random.RandomState(np_state_list[rank])
    device = torch.device(f'cuda:{rank}')

    torch0 = torch.Tensor(np_state.randn(3).astype(np.float32)).to(device)
    group = torch.distributed.new_group(list(range(world_size)))
    torch.distributed.all_reduce(torch0, op=torch.distributed.ReduceOp.SUM, group=group) #default value for group is okay

    ret_ = sum(np.random.RandomState(x).randn(3).astype(np.float32) for x in np_state_list)
    ret0 = torch0.detach().to('cpu').numpy()
    assert hfe(ret_, ret0) < 1e-4

    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()

    # see https://pytorch.org/docs/stable/nn.html#distributeddataparallel
    torch.multiprocessing.set_start_method('spawn')

    world_size = 2
    process_list = []
    for rank in range(world_size):
        p = torch.multiprocessing.Process(target=demo_send_recv, args=(rank, world_size))
        p.start()
        process_list.append(p)
    for p in process_list:
        p.join()
        p.close()

    world_size = 2
    torch.multiprocessing.spawn(demo_send_recv, args=(world_size,), nprocs=world_size, join=True)

    world_size = 2
    torch.multiprocessing.spawn(demo_non_blocking_send_recv, args=(world_size,), nprocs=world_size, join=True)

    world_size = 2
    torch.multiprocessing.spawn(demo_tensor_sync_gradient, args=(world_size,), nprocs=world_size, join=True)

    if n_gpus>=4:
        world_size = 4
        torch.multiprocessing.spawn(demo_all_reduce_basic, args=(world_size,), nprocs=world_size, join=True)

    if n_gpus>=4:
        world_size = 4
        np_state_list = np.random.randint(0, 1000, size=world_size)
        torch.multiprocessing.spawn(demo_all_reduce, args=(np_state_list,), nprocs=world_size, join=True)
