#include <iostream>
#include <string>
#include <stdexcept>

int hf0(int id)
{
    if (id == 0)
        return 233;
    throw std::invalid_argument("id");
}

// g++ draft_exception.cpp -std=c++17 -o tbd00.exe
int main(int argc, char const *argv[])
{
    std::cout << "# draft_exception.cpp" << std::endl;

    std::cout << hf0(0) << std::endl;
    // std::cout << hf0(1) << std::endl; //error

    return 0;
}
