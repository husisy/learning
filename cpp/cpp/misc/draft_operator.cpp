#include <iostream>
#include <string>

void test_shift()
{
    std::cout << "\ntest_shift\n";

    std::cout << "233<<1: " << (233<<1) << std::endl;
    std::cout << "234<<1: " << (234<<1) << std::endl;
    std::cout << "233<<2: " << (233<<2) << std::endl;
    std::cout << "234<<2: " << (234<<2) << std::endl;
}


// g++ draft_operator.cpp -std=c++17 -o tbd00.exe
int main(int argc, char *argv[])
{
    std::cout << "# draft_shift.cpp" << std::endl;

    test_shift();

    std::cout << std::endl;
    return 0;
}
