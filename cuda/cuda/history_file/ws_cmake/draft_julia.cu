#include <iostream>
#include <cassert>
#include <fstream>

void julia_cpu_kernel(double xmin, double xmax, int xdim, double ymin, double ymax, int ydim,
            double c_real, double c_imag, bool* ret0)
{
    int num_iteration = 200;
    for (int ind0=0; ind0<xdim; ind0++)
    {
        for (int ind1=0; ind1<ydim; ind1++)
        {
            double z_real = xmin + (xmax-xmin)*(ind0/((double)xdim));
            double z_imag = ymin + (ymax-ymin)*(ind1/((double)ydim));
            double tmp0, tmp1;
            for (int ind2=0; ind2<num_iteration; ind2++)
            {
                tmp0 = z_real*z_real - z_imag*z_imag + c_real;
                tmp1 = 2*z_real*z_imag + c_imag;
                z_real = tmp0;
                z_imag = tmp1;
                if ((z_real*z_real + z_imag*z_imag) >= 1000)
                {
                    break;
                }
            }
            ret0[ind0*ydim + ind1] = (z_real*z_real + z_imag*z_imag) < 1000;
        }
    }
    return;
}

__global__ void julia_gpu_kernel(double xmin, double xmax, int xdim, double ymin, double ymax, int ydim,
            double c_real, double c_imag, bool* ret0)
{
    int num_iteration = 200;
    int ind0=blockIdx.x, ind1=blockIdx.y;
    double z_real = xmin + (xmax-xmin)*(ind0/((double)xdim));
    double z_imag = ymin + (ymax-ymin)*(ind1/((double)ydim));
    double tmp0, tmp1;
    for (int ind2=0; ind2<num_iteration; ind2++)
    {
        tmp0 = z_real*z_real - z_imag*z_imag + c_real;
        tmp1 = 2*z_real*z_imag + c_imag;
        z_real = tmp0;
        z_imag = tmp1;
        if ((z_real*z_real + z_imag*z_imag) >= 1000)
        {
            break;
        }
    }
    ret0[ind0*ydim + ind1] = (z_real*z_real + z_imag*z_imag) < 1000;
}


// nvcc draft_julia.cu -o tbd00.exe
// nvcc draft_julia.cu -o tbd00.exe -ccbin "C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/bin"
int main(int argc, char const *argv[])
{
    int xdim=512, ydim=512;
    double xmin=-1.5, xmax=1.5, ymin=-1.5, ymax=1.5, c_real=-0.8, c_imag=0.156;
    bool* bitmap_cpu = new bool[xdim*ydim];
    bool* bitmap_gpu = new bool[xdim*ydim];

    julia_cpu_kernel(xmin, xmax, xdim, ymin, ymax, ydim, c_real, c_imag, bitmap_cpu);

    bool* bitmap_gpu_d;
    cudaMalloc((void **)&bitmap_gpu_d, xdim*ydim*sizeof(bool));
    julia_gpu_kernel<<<dim3(xdim,ydim), 1>>>(xmin, xmax, xdim, ymin, ymax, ydim, c_real, c_imag, bitmap_gpu_d);
    cudaMemcpy(bitmap_gpu, bitmap_gpu_d, xdim*ydim*sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(bitmap_gpu_d);

    std::ofstream fid("tbd00.txt", std::ios::binary);
    // run draft_julia.py to see the figure
    for (int ind0=0; ind0<xdim; ind0++)
    {
        for (int ind1=0; ind1<ydim; ind1++)
        {
            assert(bitmap_cpu[ind0*ydim+ind1]==bitmap_gpu[ind0*ydim+ind1]);
            fid << bitmap_cpu[ind0*ydim+ind1];
        }
        fid << "\n";
    }
    fid.close();
    delete[] bitmap_cpu, bitmap_gpu;
    return 0;
}
