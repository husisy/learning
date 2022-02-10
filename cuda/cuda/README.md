# cuda

1. link
   * [github/parallel101](https://github.com/parallel101/course)
   * [github/nvidia/cuda-samples](https://github.com/NVIDIA/cuda-samples) `helper_cuda.h` `helper_string.h`
   * [GPU高性能编程CUDA实战](https://book.douban.com/subject/5917267//)，[CUDA by example](https://developer.nvidia.com/cuda-example)
   * [知乎-CUDA入门与C混合编程](https://zhuanlan.zhihu.com/p/65241024)
   * [CUDA tutorial](https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/)
   * [nvidia documentation](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html)
   * [nvidia-developer-blog/an-even-easier-introduction-to-CUDA](https://developer.nvidia.com/blog/even-easier-introduction-cuda)
2. concept
   * kernel
   * memory: thread-local, thread-shared, global, constant, texture
   * CUDA instruction set architecture (PTX) PTX code: an assembly form
   * cubin object: binary form
   * unified memory
   * SIMT scheduling, independent thread scheduling
   * cuda runtime
   * grid stride loops
3. 偏见
   * 区分`1.0f`和`1.0`，前者是`float`，后者是`double`
   * 区分`sinf()`和`sin()`，前者是`float`签名的函数，后者是`double`
   * 在GPU硬件上，`opencl`可能比`cuda`慢一倍
4. `cudaDeviceProp` [CUDA-doc](https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html)
5. compute capability: Volta7, Pascal6, Maxwell5, Kepler3, Fermi2, Tesla1
6. nvcc option
   * binary compatibility: `-code=sm_35`
   * PTX compatibility: `-arch=compute_30`
   * 64-bit `-m64 -m32`

float function

```txt
sinf sqrtf rsqrtf cbrtf rcbrtf powf sinf cosf sinpinf cospif sincosf sincospif
logf log2f log10f expf exp2f exp10f tanf atanf asinf acosf fmodf fabsf fminf fmaxf

__sinf __expf __logf __cosf __powf __fdividef
```

## minimum working example

`draft00.cu`

```c
#include<stdio.h>
__global__ void hf0()
{
    printf("hello world for GPU\n");
}
int main()
{
    mykernel<<<1, 2>>>();
    hf0();
    return 0;
}
```

`CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
project(draft00 LANGUAGES CXX CUDA)
add_executable(draft00.exe draft00.cu)
```

1. compile
   * linux: `nvcc -o tbd00.exe draft00.cu`
   * windows (recommand to use `cmake` instead): `nvcc -o tbd00.exe draft00.cu -ccbin "C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/bin"`
2. compile with `cmake`
   * `mkdir build && cd build`
   * `cmake -S .. -B .`
   * `cmake --build .`
3. run `./tbd00.exe`
