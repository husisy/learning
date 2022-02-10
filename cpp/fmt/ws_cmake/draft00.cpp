#include <iostream>
#include <string>

#include "fmt/format.h"

int main(int argc, char const *argv[])
{
    std::cout << fmt::format("hello {}.", "world") << std::endl;
    return 0;
}
