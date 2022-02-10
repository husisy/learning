#include <iostream>
#include <tuple>

std::tuple<int, double, std::string> hf0()
{
    return std::make_tuple(1, 2.3, "456");
}

int main()
{
    auto [x0, x1, x2] = hf0();
    std::cout << x0 << ", " << x1 << ", " << x2 << std::endl;
    return 0;
}