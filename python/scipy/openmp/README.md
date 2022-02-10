# openmp, scipy性能

scipy使用了openmp来进行多线程并行，对于这一特性进行一系列测试

1. link
   * [scipy-cookbook](https://scipy-cookbook.readthedocs.io/items/ParallelProgramming.html)
2. 并行机制
3. 关键词：
   * BLAS (basic linear algebra subroutine), ATLAS, MKL, GOTO, OpenMP

```bash
export OMP_NUM_THREADS=1
os.environ['OMP_NUM_THREADS'] = '1'
# https://github.com/ME-ICA/tedana/issues/188
# https://github.com/numpy/numpy/issues/11826
# https://github.com/RasmussenLab/vamb/issues/45
```

---

计算量为两个`(N,N)`矩阵的矩阵乘，重复`5`次，开启多线程的提速比（以单线程为基准）

| matrix-size | 64 | 256 | 1024 | 4096 |
| :-: | :-: | :-: | :-: | :-: |
| thread=1 | `1.0` | `1.0` | `1.0` | `1.0` |
| thread=2 | `0.975` | `1.85` | `1.93` | `1.95` |
| thread=4 | `1.06` | `2.91` | `3.22` | `3.66` |
| thread=8 | `1.03` | `3.44` | `4.81` | `5.57` |
| thread=16 | `0.547` | `2.81` | `4.44` | `8.23` |
| thread=24 | `1.03` | `3.08` | `4.97` | `8.9` |
| thread=32 | `0.964` | `2.08` | `5.04` | `8.94` |

---

计算量为两个`(2048,2048)`矩阵的矩阵乘，总共重复`192`次，分为多个进程（计算量平分，进程间完全无通信），每个进程进而开启多个线程，用时情况如下表，sync-time表示计时前后有一次同步（考虑到不同进程速度差异较大）

| `NumProcess x NumThread` | average-time (s) | sync-time (s) |
| :-: | :-: | :-: |
| `1x24` | `9.01` | `9.01` |
| `2x12` | `7.89` | `8.08` |
| `4x6` | `7.98` | `8.16` |
| `8x3` | `7.34` | `7.81` |
| `12x2` | `7.53` | `7.87` |
| `24x1` | `6.66` | `7.28` |

---

计算量为两个`(2048,2048)`矩阵的矩阵乘，每个进程重复`192`次，用时情况如下表

| `NumProcess x NumThread` | average-time (s) | sync-time (s) |
| :-: | :-: | :-: |
| `1x1` | `78.8` | `78.8` |
| `24x1` | `141` | `143` |

---

计算量为`(2048,2048)`矩阵的`sin`，每个进程重复`192`次，用时情况如下表

| `NumProcess x NumThread` | average-time (s) | sync-time (s) |
| :-: | :-: | :-: |
| `1x1` | `10.9` | `10.9` |
| `24x1` | `13.4` | `13.9` |
