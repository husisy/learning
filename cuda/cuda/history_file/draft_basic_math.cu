#include <cstdio>
#include <cassert>
#include <cstdlib> //rand() RAND_MAX

__global__ void kernel_demo_add(int a, int b, int *c)
{
    *c = a + b;
}

void demo_add()
{
    printf("\n# demo_add\n");
    int c;
    int *device_c;
    cudaMalloc((void**)&device_c, sizeof(int));
    kernel_demo_add<<<1, 1>>>(2, 7, device_c);
    cudaMemcpy(&c, device_c, sizeof(int), cudaMemcpyDeviceToHost);
    printf("2+7=%d\n", c);
    cudaFree(device_c);
}

__global__ void kernel_demo_vector_add(const float *A, const float *B, float *C, int num_element)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < num_element)
    {
        C[i] = A[i] + B[i];
    }
}

void demo_vector_add(int num_element=5000)
{
    size_t size = num_element * sizeof(float);

    float *h_A = (float *)malloc(size); //vector A in host
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    for (int i = 0; i < num_element; ++i)
    {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }

    float *d_A = NULL; //vector A in device
    float *d_B = NULL;
    float *d_C = NULL;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (num_element + threadsPerBlock - 1) / threadsPerBlock; //equivalent to ceil

    kernel_demo_vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, num_element);
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < num_element; ++i)
    {
        assert(fabs(h_A[i] + h_B[i] - h_C[i]) < 1e-5);
    }
    printf("\n# demo_vector_add: passed\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
}

// nvcc draft_basic_math.cu -o tbd00.exe
// nvcc draft_basic_math.cu -o tbd00.exe -ccbin "C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/bin"
int main()
{
    demo_add();
    demo_vector_add();
    return 0;
}
