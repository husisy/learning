# apex

1. link
   * [github / nvidia / apex](https://github.com/NVIDIA/apex)
   * [apex documentation](https://nvidia.github.io/apex/amp.html)
   * [github / nvidia / apex / example / imagenet](https://github.com/NVIDIA/apex/tree/master/examples/imagenet)
   * [nvidia cuda documentation / mixed precision training](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)
2. concept
   * GEMMs: generalized (dense) matrix-matrix multiplies
3. `opt_level`
   * `O0`: FP32 training，用于建立accuracy baseline
   * `O1`: mixed precision，大多数场景下推荐使用
   * `O2`: almost FP16 mixed precisioin
   * `O3`: FP16 training，用于建立speed baseline

install

```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .

# for fairseq
export CUDA_HOME=/usr/local/cuda-11.1
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" .
```

TODO

1. run github imagenet example
2. apex专门写了optimizer部分的代码，KFAC optimizer未必和apex兼容

## installation

1. link
   * [博客-在conda安装的cuda环境中安装apex](http://ws.nju.edu.cn/blog/2019/10/%E5%9C%A8conda%E5%AE%89%E8%A3%85%E7%9A%84cuda%E7%8E%AF%E5%A2%83%E4%B8%AD%E5%AE%89%E8%A3%85apex/)

```bash
conda create -y -n cuda110-a
conda install -y -n cuda110-a -c conda-forge cudatoolkit-dev=11.0
conda install -y -n cuda110-a -c pytorch pytorch torchvision torchaudio
conda install -y -n cuda110-a -c conda-forge cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow protobuf scipy requests tqdm lxml opt_einsum pylint

pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
# some contrib required
# pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--fast_multihead_attn" .
```
