#include <cmath>

// g++ -fopenmp draft00.cpp -o tbd00.exe
int main()
{
    const int size = 256;
    double sinTable[size];

    // multiple threads
#pragma omp parallel for
    for (int x = 0; x < size; ++x)
    {
        sinTable[x] = std::sin(2 * M_PI * x / size);
    }
}
