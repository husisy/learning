#include <iostream>
#include <nlohmann/json.hpp>

// g++ draft00.cpp -std=c++11 -o tbd00.exe -I $env:CPP_THIRD_PARTY_INCLUDE
int main(int argc, char const *argv[])
{
    nlohmann::json z0;
    z0["pi"] = 3.1415926535;
    z0["happy"] = true;
    z0["name"] = "Niels";
    std::cout << z0;
    return 0;
}
