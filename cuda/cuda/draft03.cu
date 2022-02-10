#include <cstdio>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include <vector>

template <class T>
struct CudaAllocator
{
    using value_type = T;

    T *allocate(size_t size)
    {
        T *ptr = nullptr;
        checkCudaErrors(cudaMallocManaged(&ptr, size * sizeof(T)));
        return ptr;
    }

    void deallocate(T *ptr, size_t size = 0)
    {
        checkCudaErrors(cudaFree(ptr));
    }

    template <class... Args>
    void construct(T *p, Args &&...args)
    {
        if constexpr (!(sizeof...(Args) == 0 && std::is_pod<T>::value))
            ::new ((void *)p) T(std::forward<Args>(args)...);
    }
};

__global__ void demo00_hf0(int *arr, int n)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        arr[i] = i;
    }
}

void demo00_vector()
{
    printf("\n# demo00_vector\n");
    int n = 1022;
    int ret_ = (n * (n - 1)) / 2;
    int ret0;
    std::vector<int, CudaAllocator<int>> arr(n);

    demo00_hf0<<<2, 128>>>(arr.data(), n);
    checkCudaErrors(cudaDeviceSynchronize());
    ret0 = 0;
    for (int i = 0; i < n; i++)
    {
        ret0 += arr[i];
    }
    printf("sum[n=%d]: ret_=%d, ret0=%d\n", n, ret_, ret0);
}

template <class Func>
__global__ void parallel_for_naive(int *arr, int n, Func func)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        arr[i] = func(i);
    }
}

template <class Func>
__global__ void parallel_for(int n, Func func)
{
    for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        func(i);
    }
}

struct MyFunctor01
{
    __device__ int operator()(int i) const
    {
        return -i;
    }
};

void demo01_functor()
{
    printf("\n# demo01_functor\n");
    int n = 8191;
    int ret_ = -(n * (n - 1)) / 2;
    int ret0, ret1, ret2;

    int *arr0;
    checkCudaErrors(cudaMallocManaged(&arr0, n * sizeof(int)));
    parallel_for_naive<<<2, 128>>>(arr0, n, MyFunctor01{});
    checkCudaErrors(cudaDeviceSynchronize());
    ret0 = 0;
    for (int i = 0; i < n; i++)
    {
        ret0 += arr0[i];
    }
    cudaFree(arr0);

    std::vector<int, CudaAllocator<int>> arr1(n);
    int *arr1_data = arr1.data();
    parallel_for<<<2, 128>>>(n, [=] __device__(int i)
                             { arr1_data[i] = -i; });
    checkCudaErrors(cudaDeviceSynchronize());
    ret1 = 0;
    for (int i = 0; i < n; i++)
    {
        ret1 += arr1[i];
    }

    std::vector<int, CudaAllocator<int>> arr2(n);
    parallel_for<<<32, 128>>>(n, [arr2 = arr2.data()] __device__(int i)
                              { arr2[i] = -i; });
    checkCudaErrors(cudaDeviceSynchronize());
    ret2 = 0;
    for (int i = 0; i < n; i++)
    {
        ret2 += arr2[i];
    }

    printf("sum[n=%d]: ret_=%d, ret0=%d, ret1=%d, ret2=%d\n", n, ret_, ret0, ret1, ret2);
}

int main()
{
    demo00_vector();

    demo01_functor();

    return 0;
}
