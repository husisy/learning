# Numba

1. link
   * [github](https://github.com/numba/numba)
   * [official-site](http://numba.pydata.org/)
   * [documentation](https://numba.readthedocs.io/en/stable/index.html)
2. install
   * `conda install -c conda-forge numba`
   * `pip install numba`
3. check system info `numba -s`
4. mode: `nopython`, `object`
5. modes of operation
   * eager, decoration-time compilation
   * lazy, call-time compilation, dynamic universal functions: **禁止**使用
6. 偏见
   * **禁止**使用`object` mode
   * **禁止**使用lazy compilation模式
7. 目标平台选择
   * `cpu`: 单线程cpu，额外开销overhead最小，适合`<1KB`大小数组
   * `parallel`: multi-core cpu，适合`<1MB`大小数组
   * `cuda`: cuda-gpu，适合`>1MB`数组，数据往返GPU的搬运开销不可忽略
