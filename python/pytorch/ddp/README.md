# multi GPU

1. link
   * PyTorch Distributed: Experiences on Accelerating Data Parallel Training, see [arxiv](https://arxiv.org/abs/2006.15704)
   * [pytorch notes / CUDA semantics](https://pytorch.org/docs/stable/notes/cuda.html) DONE
   * [documentation / DDP design note](https://pytorch.org/docs/master/notes/ddp.html) DONE
   * [documentation / multiprocessing best practices](https://pytorch.org/docs/stable/notes/multiprocessing.html) DONE but not understood
   * [github / pytorch examples](https://github.com/pytorch/examples/tree/master/distributed) TODO
   * [github / pytorch examples / imagenet](https://github.com/pytorch/examples/tree/master/imagenet) TODO
   * [documentation / DDP](https://pytorch.org/docs/stable/nn.html#distributeddataparallel) DONE
   * [documentation / torch.distributed](https://pytorch.org/docs/stable/distributed.html) DONE
   * [Seba/an introduction to distributed deep learning](http://seba1511.net/dist_blog/) DONE
   * [documentation / getting started with distributed data parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) DONE
   * [documentation / writing distributed applications with pytorch](https://pytorch.org/tutorials/intermediate/dist_tuto.html) DONE
   * [tutorial / pytorch RPC](https://pytorch.org/tutorials/intermediate/rpc_tutorial.html) 暂且不考虑RPC
   * [documentation / single machine model parallel best practices](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html) 不考虑model parallelism
   * [documentation / multi-GPU examples](https://pytorch.org/tutorials/beginner/former_torchies/parallelism_tutorial.html)不考虑model parallelism
   * [documentation / rpc](https://pytorch.org/docs/stable/rpc.html) TODO 暂不考虑RPC
   * filelock with downloading data, see [ray/documentation](https://docs.ray.io/en/master/using-ray-with-pytorch.html) TODO
2. 偏见
   * 使用`torch.nn.parallel.DistributedDataParallel`，而不使用`multiprocessing`，而不使用`torch.nn.DataParallel`, see [pytorch notes](https://pytorch.org/docs/stable/notes/cuda.html#use-nn-parallel-distributeddataparallel-instead-of-multiprocessing-or-nn-dataparallel) and [pytorch rpc](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#comparison-between-dataparallel-and-distributeddataparallel)
3. environment variable
   * `CUDA_VISIBLE_DEVICES`
   * `CUDA_LAUNCH_BLOCKING`: close asynchronous execution
   * `MASTER_PORT`：`rank=0`进程会监听该端口
   * `MASTER_ADDR`：`rank=0`进程所在地址，其余进程会与该地址通信
   * `WORLD_SIZE`：总进程数，`rank=0`进程需要该参数用于确认需要等待多少个进程
   * `RANK`：每个进程需要该参数确认自己是MASTER还是WORKER
4. cluster coordination tool
   * [pdsh](https://linux.die.net/man/1/pdsh)
   * [clustershell](https://cea-hpc.github.io/clustershell/)
   * [slurm](https://slurm.schedmd.com/)
5. concept
   * data parallelism, model parallelism（不适用）
   * tree-reductions, parameter server（不适用）
   * synchronous, asynchronous
6. 偏见
   * DDP特指`torch.nn.parallel.DistributedDataParallel`，**禁止**使用`torch.nn.DataParallel`
   * 不考虑model parallelism（个人偏见）
   * 使用`torch.distributed.launch`，**不建议**使用`torch.multiprocessing.spawn`，前者便于统一串行与并行代码
7. DDP
   * DDP中的多进程**禁止**共享GPU设备（会造成死锁），每个进程可以占用多个GPU设备（即model parallel）
   * DDP确保不同进程中的模型初始化一致
   * DDP中的同步点：the constructor, the forward pass, the backward pass。这些同步点必须有序且等量
   * 「balance the workloads distributions across processes」是用户的责任而非DDP的责任
   * skewed processing speeds可能会导致`TimeoutError`，默认参数`init_process_group(timeout=1800)`
   * 一个进程负责save the model。如果其余进程需要加载模型务必先`dist.barrier()`确保模型保存完毕
   * graidents-communication与back backpropagation异步进行
   * 注意`torch.load(map_localtion=)`
   * 同步序列必须精确一致
   * `bn.running_mean/bn.running_var`在正向传播之前同步
8. different reduction algorithm work best with different network topologies: ring, butterfly, slimfly, ring-segmented
9. backend选择原则
   * nccl not support send-recv yet, see [pytorch/backends](https://pytorch.org/docs/stable/distributed.html#backends)
   * pytorch分布式目前仅支持linux
   * gloo and nccl backends are built and included, mpi backends requires build pytorch from source
   * 当完全GPU training时使用nccl backend，当完全CPU training时使用gloo backend
   * GPU hosts with InfiniBand interconnect时使用nccl backend（唯一支持）
   * GPU hosts with Ethernet interconnect时使用nccl backend（性能最优）
   * CPU hosts with InfiniBand interconnect时使用gloo backend
   * CPU hosts with Ethernet interconnect时使用gloo backend unless you have specific reasons to use MPI
10. `torch.multiprocessing`语义
    * tensor的grad也会一起shared....暂不打算使用底层queue语义
11. pytorch RPC framework `torch.distributed.rpc`
    * 使用场景：parameter server, reinforcement learning
    * `torch.distributed.rpc.remote()`, `torch.distributed.rpc.rpc_sync()`, `torch.distributed.rpc.RRef`
    * distributed autograd context
    * 暂且不考虑RPC

pytorch ddp setup

```Python
# os.environ['MASTER_ADDR'] = '127.0.0.1' #localhost
# os.environ['MASTER_PORT'] = '23333'
# torch.distributed.init_process_group(backend='nccl', rank=rank, world_size=world_size)

torch.distributed.init_process_group(backend='nccl',
      init_method='tcp://127.0.0.1:23333', rank=rank, world_size=world_size)

torch.distributed.get_backend()
torch.distributed.get_world_size()
torch.distributed.get_rank()
```

`torch.distributed.launch`会添加如下环境变量

| key | origin | proc-i |
| :-: | :-: | :-: |
| `OMP_NUM_THREADS` | `None` | `1` |
| `LOCAL_RANK` | `None` | `i` |
| `WORLD_SIZE` | `None` | `N` |
| `MASTER_ADDR` | `None` | `127.0.0.1` |
| `MASTER_PORT` | `None` | `29500` |
| `RANK` | `None` | `i` |

TODO

1. `dist.destroy_process_group()`
2. 集群cluster
3. see `PCL-hsy/project/pytorch-example-imagenet/draft01.py`
4. should we synchronize weight every several epochs to avoid numerical errors
5. check graidient sync
6. quantized gradients
7. divide the tensor into chunks
