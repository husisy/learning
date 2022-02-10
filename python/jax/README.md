# jax

1. link
   * [github](https://github.com/google/jax)
   * [documentation](https://jax.readthedocs.io/en/latest/)
   * [jaxlib-whl](https://storage.googleapis.com/jax-releases/jax_releases.html)
2. 安装
   * `conda install -c conda-forge jax jaxlib`
   * cpu-only: `pip install jax jaxlib`
   * gpu: `pip install jax jaxlib==0.1.61+cuda112 -f https://storage.googleapis.com/jax-releases/jax_releases.html`
3. XLA
4. `jax.jit`表达能力
   * boolean indexing没戏，shape改变
   * static or traced operation：用numpy来操作静态信息
   * 禁止在任何`jax`函数中以及控制流原语（例如`jax.lax.scan, jax.lax.fori_loop`）中使用Python iterators
5. 近似在位操作的函数：`jax.ops.index_update`, `jax.ops.index_add`, `jax.ops.index_min`, `jax.ops.index_max`, `jax.ops.index`。当原内存空间的变量未使用时，则会进行在位操作
6. jax中的越界访问视作为定义的行为，不能假定其行为
7. startup config
   * `jax_debug_nans`
   * `jax_enable_x64`
8. Model-Agnostic Meta-Learning (MAML)
9. pytree, nest, tree

```bash
conda create -n jax-cuda112
conda install -y -n jax-cuda112 -c conda-forge cython ipython pytest matplotlib h5py pandas pylint jupyterlab pillow protobuf scipy requests tqdm lxml opt_einsum
conda activate jax-cuda112
pip install jax jaxlib==0.1.68+cuda111 -f https://storage.googleapis.com/jax-releases/jax_releases.html
# export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/cuda-11.2/lib64:/usr/local/cuda-11.2/extras/CUPTI/lib64"
```
