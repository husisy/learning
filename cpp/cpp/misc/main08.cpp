#include <iostream>
#include <vector>
#include <algorithm>

int main()
{
    std::vector<int> vec{1, 2, 3, 4};

    if (const std::vector<int>::iterator ind1 = std::find(vec.begin(), vec.end(), 2);
        ind1 != vec.end())
    {
        *ind1 = 4;
    }

    for(std::vector<int>::iterator x = vec.begin(); x!=vec.end(); ++x){
        std::cout << *x << std::endl;
    }

    return 0;
}