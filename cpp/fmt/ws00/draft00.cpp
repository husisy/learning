#include <iostream>
#include <string>

#include "fmt/format.h"

// g++ draft00.cpp src/format.cc -std=c++11 -I ./include -o tbd00.exe
int main(int argc, char const *argv[])
{
    std::cout << fmt::format("hello {}.", "world") << std::endl;
    return 0;
}
