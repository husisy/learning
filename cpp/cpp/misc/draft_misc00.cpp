#include <iostream>
#include <string>

void test_enum()
{
    std::cout << "\ntest_enum\n";
    enum
    {
        x0,
        x1,
        x2 = x1 + 2
    };
    std::cout << "x0: " << x0 << std::endl;
    std::cout << "x1: " << x1 << std::endl;
    std::cout << "x2: " << x2 << std::endl;
}

// g++ draft_misc00.cpp -std=c++17 -o tbd00.exe
// ./tbd00.exe 2 23 233
int main(int argc, char *argv[])
{
    std::cout << "# draft02.cpp" << std::endl;

    std::cout << "\n# test argc / argv" << std::endl;
    std::cout << "argc: " << argc << std::endl;
    for (int i = 0; i < argc; i++)
        std::cout << "argv[" << i << "]: " << argv[i] << std::endl;

    test_enum();

    std::cout << std::endl;
    return 0;
}
