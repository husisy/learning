#include <cstdio>
#include <iostream>

__global__ void kernel_demo_printf()
{
    printf("hello world from GPU\n");
    // "std::cout" will fail
}


void demo_printf()
{
    std::cout << "\n# demo_cout\n";
    kernel_demo_printf<<<1, 2>>>();
    cudaDeviceSynchronize();
}

void demo_handle_error()
{
    // TODO
    // see cuda_by_example/common/book.h
}

// nvcc draft_misc.cu -o tbd00.exe
// nvcc draft_misc.cu -o tbd00.exe -ccbin "C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/bin"
int main()
{
    demo_printf();
    return 0;
}
