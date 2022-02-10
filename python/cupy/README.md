# cupy

1. link
   * [github](https://github.com/cupy/cupy)
   * [documentation](https://docs-cupy.chainer.org/en/stable/overview.html)
2. install
   * `pip install cupy-cuda100`, `pip install cupy-cuda101`, `pip install cupy-cuda102`
   * `conda install -c conda-forge cupy`
   * `conda install -c conda-forge cudatoolkit nccl`
3. `cupy.cuda.nccl`
   * `export NCCL_SOCKET_IFNAME=eth0`，支持节点间通信，见`demo_inter_nccl.py`
   * `cp.cuda.nccl.NcclCommunicator()`需放置于`cp.cuda.Device()`上下文中，否则会抛出`NCCL_ERROR_INVALID_USAGE` [github/cupy-issue5300](https://github.com/cupy/cupy/issues/5300)
   * `unique_id`与进程绑定，对于`rank=0`，不可以在进程0生成再交给进程1使用，见`demo_nccl.py`
