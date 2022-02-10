#include <iostream>
#include <vector>
#include <string>
#include <cassert>

#include "zcutils.cpp"

void test_range()
{
    std::cout << "\n# test_range\n";

    std::cout << "range(3): " << range(3) << std::endl;
    std::cout << "range(0): " << range(0) << std::endl;
    std::cout << "range(-1): " << range(-1) << std::endl;

    std::cout << "range(1,3): " << range(1,3) << std::endl;
    std::cout << "range(1,1): " << range(1,1) << std::endl;
    std::cout << "range(1,0): " << range(1,0) << std::endl;

    std::cout << "range(0,10,3): " << range(0,10,3) << std::endl;
    std::cout << "range(0,10,-3): " << range(0,10,-3) << std::endl;
    std::cout << "range(0,-10,3): " << range(0,-10,3) << std::endl;
    std::cout << "range(0,-10,-3): " << range(0,-10,-3) << std::endl;
}


// g++ draft_misc.cpp -std=c++11 -o tbd00.exe
int main(int argc, char const *argv[])
{
    std::cout << "draft_misc.cpp\n";

    test_range();

    return 0;
}
