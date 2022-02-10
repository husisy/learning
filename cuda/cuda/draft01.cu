#include <cstdio>
#include <cuda_runtime.h>

__global__ void demo00()
{
    unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int tnum = blockDim.x * gridDim.x;
    printf("\n# demo00 [%d/%d] blockIdx.x=%d, gridDim.x=%d, threadIdx.x=%d, blockDim.x=%d\n",
           tid, tnum, blockIdx.x, gridDim.x, threadIdx.x, blockDim.x);
}

__global__ void demo01()
{
    unsigned int tid = blockIdx.z*gridDim.y + blockIdx.y;
    tid = tid*gridDim.x + blockIdx.x;
    tid = tid*blockDim.z + threadIdx.z;
    tid = tid*blockDim.y + threadIdx.y;
    tid = tid*blockDim.x + threadIdx.x;
    unsigned int tnum = blockDim.x * blockDim.y * blockDim.z * gridDim.x * gridDim.y * gridDim.z;
    printf("\n# demo01 [%d/%d] blockIdx=(%d,%d,%d), gridDim=(%d,%d,%d), "
           "threadIdx=(%d,%d,%d), blockDim=(%d,%d,%d)\n",
           tid, tnum, blockIdx.x, blockIdx.y, blockIdx.z,
           gridDim.x, gridDim.y, gridDim.z,
           threadIdx.x, threadIdx.y, threadIdx.z,
           blockDim.x, blockDim.y, blockDim.z);
}

__global__ void demo02_hf0(int x) {
    printf("demo02_hf0[%d][%d/%d]\n", x, threadIdx.x, blockDim.x);
}

//set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
__global__ void demo02() {
    printf("\n# demo02[%d/%d]\n", threadIdx.x, blockDim.x);
    int numthreads = threadIdx.x * threadIdx.x + 1;
    demo02_hf0<<<1, numthreads>>>(threadIdx.x);
}


// nvcc draft01.cu -o tbd00.exe
int main()
{
    demo00<<<2, 3>>>(); //gridDim, blockDim
    cudaDeviceSynchronize();

    demo01<<<dim3(2, 1, 1), dim3(2, 2, 2)>>>();
    cudaDeviceSynchronize();

    demo02<<<1, 3>>>();
    cudaDeviceSynchronize();
    return 0;
}
