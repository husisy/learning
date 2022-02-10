#include <cstdio>
#include <cmath>
#include <cassert>

__global__ void cuda_vector_add_kernel(const double *A, const double *B, double *C, int num_element)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_element)
    {
        C[i] = A[i] + B[i];
    }
}

void cuda_vector_add(const double* data0, const double* data1, int num_element, double* ret)
{
    assert(num_element>0);
    size_t size_byte = sizeof(double)*num_element;
    double *data0_d, *data1_d, *ret_d;
    cudaMalloc(&data0_d, size_byte); //TODO error check
    cudaMalloc(&data1_d, size_byte);
    cudaMalloc(&ret_d, size_byte);
    cudaMemcpy(data0_d, data0, size_byte, cudaMemcpyHostToDevice);
    cudaMemcpy(data1_d, data1, size_byte, cudaMemcpyHostToDevice);
    int threadsPerBlock = 256; //TODO how to select a suitable threads
    int blocksPerGrid = (num_element + threadsPerBlock - 1) / threadsPerBlock; //equivalent to ceil
    cuda_vector_add_kernel<<<blocksPerGrid, threadsPerBlock>>>(data0_d, data1_d, ret_d, num_element);
    cudaMemcpy(ret, ret_d, size_byte, cudaMemcpyDeviceToHost);
    cudaFree(data0_d);
    cudaFree(data1_d);
    cudaFree(ret_d);
}

double* cuda_vector_add(const double* data0, const double* data1, int num_element)
{
    assert(num_element>0);
    double *ret = new double[num_element];
    cuda_vector_add(data0, data1, num_element, ret);
    return ret;
}

void demo_vector_add(int num_element)
{
    double *h_A = new double[num_element];
    double *h_B = new double[num_element];
    for (int i = 0; i < num_element; ++i)
    {
        h_A[i] = rand() / (double)RAND_MAX;
        h_B[i] = rand() / (double)RAND_MAX;
    }
    double *h_C = cuda_vector_add(h_A, h_B, num_element);

    for (int i = 0; i < num_element; ++i)
    {
        assert(fabs(h_A[i] + h_B[i] - h_C[i]) < 1e-5);
    }
    printf("[zc-info] demo_vector_add() pass\n");

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
}
