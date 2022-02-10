#include <iostream>
#include <tuple>
#include <string>

std::tuple<double, std::string> hf0()
{
    return std::make_tuple(2.333, "2333");
    // return {2.33, "233"}; //c++17
}

void test_tuple()
{
    std::cout << "\n#test_tuple\n";

    // auto z0 = hf0();
    // std::cout << "tuple0: " << std::get<0>(z0) << ", "
    //           << "tuple1: " << std::get<1>(z0) << std::endl;

    double z1_f;
    std::string z1_s;
    std::tie(z1_f, z1_s) = hf0();
    std::cout << "tuple0: " << z1_f << ", "
              << "tuple1: " << z1_s << std::endl;

    // auto [z2_f, z2_s] = hf0(0); //structured binding
    // std::cout << "tuple0: " << z2_f << ", "
    //           << "tuple1: " << z2_s << std::endl;
}

// g++ draft_tuple.cpp -std=c++11 -o tbd00.exe
int main(int argc, char const *argv[])
{
    std::cout << "# draft_tuple.cpp" << std::endl;

    test_tuple();

    return 0;
}
