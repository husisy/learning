#include <cstdio>
#include <cuda_runtime.h>

__global__ void demo00()
{
    printf("\n# demo00\n");
}

__device__ __inline__ void demo01_hf0()
{
    printf("\n# demo01\n");
}

__global__ void demo01()
{
    demo01_hf0();
}

__host__ __device__ void demo02_hf0()
{
#ifdef __CUDA_ARCH__
    printf("\n# demo02 from GPU\n");
    // see https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capabilities
    printf("GPU architecture %d\n", __CUDA_ARCH__);
#else
    printf("\n# demo02 from CPU\n");
#endif
}

__global__ void demo02()
{
    demo02_hf0();
}

void demo_query_device()
{
    printf("\n# demo_query_device\n");
    int count;
    cudaDeviceProp prop;

    cudaGetDeviceCount(&count);
    for (int i = 0; i < count; i++)
    {
        cudaGetDeviceProperties(&prop, i);
        printf("## general information for device %d\n", i);
        printf("name: %s\n", prop.name);
        printf("compute capability: %d.%d\n", prop.major, prop.minor);
        printf("clock rate: %d\n", prop.clockRate);
        printf("device copy overlap: %s\n", prop.deviceOverlap ? "enabled" : "disabled");
        printf("kernel execution timeout: %s\n", prop.kernelExecTimeoutEnabled ? "enabled" : "disabled");

        printf("## memory information for device %d\n", i);
        printf("total global mem:  %ld\n", prop.totalGlobalMem);
        printf("total constant mem:  %ld\n", prop.totalConstMem);
        printf("max mem pitch:  %ld\n", prop.memPitch);
        printf("texture alignment:  %ld\n", prop.textureAlignment);

        printf("## MP Information for device %d\n", i);
        printf("Multiprocessor count: %d\n", prop.multiProcessorCount);
        printf("Shared mem per mp:  %ld\n", prop.sharedMemPerBlock);
        printf("Registers per mp:  %d\n", prop.regsPerBlock);
        printf("Threads in warp:  %d\n", prop.warpSize);
        printf("Max threads per block:  %d\n", prop.maxThreadsPerBlock);
        printf("Max thread dimensions:  (%d, %d, %d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max grid dimensions:  (%d, %d, %d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    }
}

// nvcc draft00.cu -o tbd00.exe
int main()
{
    demo_query_device();

    demo00<<<1, 1>>>();

    demo01<<<1, 1>>>();

    demo02<<<1, 1>>>(); //gpu call
    cudaDeviceSynchronize();
    demo02_hf0(); //cpu call

    return 0;
}
