#include <iostream>
#include <ctime>

void test_localtime()
{
    std::cout << "\ntest_localtime\n";
    std::time_t x0;
    std::time(&x0);
    std::cout << std::asctime(std::localtime(&x0)) << std::endl;
}

// g++ draft01_datetime.cpp -std=c++11 -o tbd00.exe
// tbd00.exe
int main(int argc, char *argv[])
{
    std::cout << "# draft01_datetime.cpp" << std::endl;

    test_localtime();

    std::cout << std::endl;
    return 0;
}
