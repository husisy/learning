#include <iostream>
#include <vector>
// #include <blas.h>

extern "C" double ddot_(int* N, double* x, int* incx, double* y, int* incy);

// see https://stackoverflow.com/q/10112135/7290857
// g++ draft_ddot.cpp -o tbd00.exe -lblas
int main(int argc, char *argv[])
{
    int N0=2^20;
    int one = 1;

    std::vector<double> a(N0, 0.0), b(N0, 0.0);
    double c=233;
    c = ddot_(&N0, &a[0], &one, &b[0], &one);
    std::cout << c << std::endl;

    return(0);
}
