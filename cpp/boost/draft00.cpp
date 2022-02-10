#include <iostream>
#include <string>

#include <boost/format.hpp>

// g++ draft00.cpp -std=c++11 -o tbd00.exe -I $env:CPP_THIRD_PARTY_INCLUDE
int main(int argc, char const *argv[])
{
    std::cout << boost::format("hello %1%") % "world\n";

    return 0;
}
