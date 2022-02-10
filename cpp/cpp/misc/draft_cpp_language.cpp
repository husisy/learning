#include <iostream>
#include <string>

void test_pointer_to_pointer(int dim0=20, int dim1=30)
{
    std::cout << "\n# test_pointer_to_pointer\n";
    int **z0;

    std::cout << "new int *[233]\n";
    z0 = new int *[dim0];
    for (int ind0=0; ind0<dim0; ind0++)
    {
        z0[ind0] = new int[dim1];
    }

    std::cout << "delete[] z0[233]\n";
    for (int ind0=0; ind0<dim0; ind0++)
    {
        delete[] z0[ind0];
    }
    delete[] z0;
}

// g++ draft_cpp_language.cpp -std=c++11 -o tbd00.exe
int main(int argc, char const *argv[])
{
    std::cout << "# draft_cpp_language.cpp\n";
    test_pointer_to_pointer();
    return 0;
}
