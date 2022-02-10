#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"

__global__ void demo00_hf0(int *result_gpu) {
    *result_gpu = 233;
}

void demo00()
{
    printf("\n# demo00\n");
    int *result_gpu;
    int result_cpu;
    checkCudaErrors(cudaMalloc(&result_gpu, sizeof(int)));
    demo00_hf0<<<1, 1>>>(result_gpu);
    // cudaMemcpy include cudaDeviceSynchronize() once
    checkCudaErrors(cudaMemcpy(&result_cpu, result_gpu, sizeof(int), cudaMemcpyDeviceToHost));
    printf("result: %d\n", result_cpu);
    cudaFree(result_gpu);
}

__global__ void demo01_hf0(int *result) {
    *result = 233;
}

void demo01_unified_memory() {
    printf("\n# demo01_unified_memory\n");
    int *result;
    checkCudaErrors(cudaMallocManaged(&result, sizeof(int)));
    demo01_hf0<<<1, 1>>>(result);
    checkCudaErrors(cudaDeviceSynchronize());
    printf("result: %d\n", *result);
    cudaFree(result);
}



__global__ void demo02_hf0(int *arr, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    arr[i] = i;
}

__global__ void demo02_hf1(int *arr, int n) {
    for (int i = blockDim.x * blockIdx.x + threadIdx.x;
         i < n; i += blockDim.x * gridDim.x) {
        arr[i] = i;
    }
}

int demo02() {
    printf("\n# demo02\n");
    int n = 1022;
    int ret0, ret1;
    int ret_ = (n*(n-1))/2; //2**16=65536

    int *arr0;
    checkCudaErrors(cudaMallocManaged(&arr0, n * sizeof(int)));
    int nthreads = 128;
    int nblocks = (n + nthreads - 1) / nthreads;
    demo02_hf0<<<nblocks, nthreads>>>(arr0, n);
    checkCudaErrors(cudaDeviceSynchronize());
    ret0 = 0;
    for (int i = 0; i < n; i++) {
        ret0 += arr0[i];
    }
    cudaFree(arr0);

    int *arr1;
    checkCudaErrors(cudaMallocManaged(&arr1, n * sizeof(int)));
    demo02_hf1<<<2, 128>>>(arr1, n);
    checkCudaErrors(cudaDeviceSynchronize());
    ret1 = 0;
    for (int i = 0; i < n; i++) {
        ret1 += arr1[i];
    }
    cudaFree(arr1);

    printf("sum[n=%d]: ret_=%d, ret0=%d, ret1=%d\n", n, ret_, ret0, ret1);

    return 0;
}


int main() {
    demo00();

    demo01_unified_memory();

    demo02();
    return 0;
}
