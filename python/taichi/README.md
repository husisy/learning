# taichi

1. link
   * [github](https://github.com/taichi-dev/taichi)
   * [documentation](https://docs.taichi.graphics/lang/articles/basic/overview)
   * [paper](http://taichi.graphics/wp-content/uploads/2019/09/taichi_lang.pdf)
   * [slide](http://taichi.graphics/wp-content/uploads/2019/12/taichi_slides.pdf)
   * [bilibili/course](https://space.bilibili.com/490448800/channel/detail?cid=154986)
2. 安装`pip install taichi`
   * `pip install taichi-tina taichi_glsl`
   * `docker pull taichidev/taichi:latest`
3. `ti example`
4. goal: performance, productivity, spatially sparse, differentiable programming, metaprogramming
5. `ti.kernel`
   * 必须type-hinted，最多支持8个参数
   * 目前@20200619仅支持返回一个scalar
6. `ti.func`
   * 仅在taichi-scope中使用
   * 不支持多个return语句
   * 目前@20200619所有`ti.func` force-inlined，不支持递归
   * `ti.func`参数passed by value
7. `ti.sync()`
8. 数据类型`i8 i16 i32 i64`, `u8 u16 u32 u64`, `f32 f64`
   * boolean type should be represented using `ti.i32`
9. 仅最外层作用域并行（不是最外层for-loop）

```bash
conda create -y -n taichi
conda install -y -n taichi -c conda-forge cudatoolkit=11.1
conda install -y -n taichi -c pytorch pytorch torchvision torchaudio
conda install -y -n taichi -c conda-forge cython ipython pytest matplotlib h5py pylint pillow protobuf scipy requests tqdm
# apt install zlib1g-dev libxi-dev libxcursor-dev libxinerama-dev libxrandr-dev libx11-dev libgl-dev libtinfo5 clang llvm
# export CXX=/usr/bin/clang
# pip install --user -r requirements_dev.txt
# python setup.py develop
```

`docker build -t zc-conda-taichi:v0 .`

```Dockerfile
FROM nvidia/cuda:11.2.1-devel-ubuntu20.04

RUN apt-get update \
    && apt-get install -y wget xauth \
    && wget -O /root/Miniconda3-latest-Linux-x86_64.sh -q "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
    && bash /root/Miniconda3-latest-Linux-x86_64.sh -p /root/miniconda3 -b \
    && . "/root/miniconda3/etc/profile.d/conda.sh" \
    && conda create -y -n taichi \
    && conda install -y -n taichi -c conda-forge cython ipython pytest matplotlib h5py pylint pillow protobuf scipy requests tqdm \
    && conda activate taichi \
    && pip install taichi cupy-cuda112 \
    && conda clean -y -a \
    && echo '. "/root/miniconda3/etc/profile.d/conda.sh"' >> /root/.bashrc \
    && echo 'conda activate taichi' >> /root/.bashrc
```
