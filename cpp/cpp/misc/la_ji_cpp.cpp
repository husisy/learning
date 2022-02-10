#include <iostream>
using namespace std;

class ZC1
{
private:
    int i;

public:
    void Hello() { std::cout << "hello" << std::endl; }
};

// g++ la_ji_cpp.cpp -std=c++17 -o tbd00.exe
int main(int argc, const char* argv[])
{
    ZC1 *ptr = nullptr;
    ptr->Hello();
    (*ptr).Hello();
    return 0;
}
