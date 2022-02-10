# tvm

1. link
   * [github](https://github.com/apache/tvm/)
   * [documentation](https://tvm.apache.org/)
   * [tutorial](https://tvm.apache.org/docs/tutorials/get_started/introduction.html)
   * [知乎-如何利用TVM快速实现超越Numpy的GEMM](https://zhuanlan.zhihu.com/p/75203171)
   * [github/netron](https://github.com/lutzroeder/netron)
   * [github/flame/how-to-optimize-gemm](https://github.com/flame/how-to-optimize-gemm)
2. install
   * TLCPack [link](https://tlcpack.ai/)
   * `conda install tlcpack -c tlcpack` not available on tsinghua-conda-mirror
   * 依赖包`pip install xgboost`
   * 依赖包 `conda install -c conda-forge xgboost onnx`
   * install from source [link](https://tvm.apache.org/docs/install/from_source.html)
3. layer
   * importer layer: tensorflow, pytorch, ONNX
   * relay, high-level IR
   * Tensor Expression (TE), Tensor Operator Inventory (TOPI)
   * AutoTVM / AutoScheduler: auto-tuning module
   * TE, Schedule: optimization specification
   * Tensor Intermediate Representation (TIR), compiler backend: LLVM, NVCC, BYOC
4. TE Scheduling Primitives: split, tile, fuse, reorder, bind, compute_at, compute_inline, compute_root [link](https://tvm.apache.org/docs/tutorials/language/schedule_primitives.html)
5. optimization
   * increase the cache hit rate of memory access
   * SIMD (Single Instruction Multi-Data), vector processing unit
6. cache, blocking
   * `dcache`: data cache
   * `icache`: instruction cache

`draft04.py`: `(2048,2048) x (2048,2048) -> (2048,2048)`

| `OMP_NUM_THREADS` | numpy | tvm |
| :-: | :-: | :-: |
| `1` | `0.205` | `0.338` |
| `24` | `0.0309` | `0.0248` |
| `-` | `0.0304` | `0.0248` |

在`OMP_NUM_THREADS=24`时，通过`htop`查看，numpy计算和核心分布乱糟糟的，而tvm仅分布在前24核

```bash
conda create -y -n cuda111-tvm
conda install -y -n cuda111-tvm -c conda-forge cudatoolkit=11.1
conda install -y -n cuda111-tvm -c conda-forge cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow protobuf scipy requests tqdm lxml pylint cudnn cmake bzip2 llvmdev=10.0

mkdir build
cp cmake/config.cmake
cd build
# nano config.cmake
cmake ..
cp ./*.so ~/mylib/tvm
export LD_LIBRARY_PATH="/home/zhangc/mylib/tvm:$LD_LIBRARY_PATH"
cd ../python
pip install -e .
```

```bash
# data
wget https://github.com/onnx/models/raw/master/vision/classification/resnet/model/resnet50-v2-7.onnx
```

```bash
tvmc compile --target "llvm" --output resnet50-v2-7-tvm.tar resnet50-v2-7.onnx
mkdir model
tar -xvf resnet50-v2-7-tvm.tar -C model
ls model
```

```bash
llc --version
```
