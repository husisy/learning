#include <iostream>
#include <vector>

void auto_example00()
{
    std::vector<int> x1{1, 2, 3, 4};
    for (std::vector<int>::iterator it0 = x1.begin(); it0 != x1.end(); it0++)
    {
        std::cout << *it0 << std::endl;
    }

    for (auto it0 = x1.begin(); it0 != x1.end(); it0++)
    {
        std::cout << *it0 << std::endl;
    }
    return;
}



int main()
{
    auto_example00();

    return 0;
}
