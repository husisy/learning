#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <set>

#include "zcutils.cpp"

void test_vector_constructor()
{
    // https://en.cppreference.com/w/cpp/container/vector/vector
    std::cout << "\n# test_vector_constructor\n";

    std::vector<std::string> z0 {"2", "23", "233"};
    std::cout << "initializer from list: " << z0 << std::endl;

    std::vector<std::string> z1(z0.begin(), z0.end());
    std::cout << "iterator constructor: " << z1 << std::endl;

    std::vector<std::string> z2(z0);
    std::cout << "copy constructor: " << z2 << std::endl;

    std::vector<std::string> z3(3, "233");
    std::cout << "repeat constructor: " << z3 << std::endl;

    std::vector<int> z4 = {2, 3};
    z4.push_back(3);;
    std::cout << ".capacity(): " << z4.capacity() << std::endl;
    std::cout << ".size(): " << z4.size() << std::endl;
    std::cout << "second elements: " << z4[1] << ", " << z4.at(1) << std::endl;
    int sum = 0;
    for (auto ind0 = z4.cbegin(); ind0 !=z4.cend(); ind0++){
        sum += *ind0;
    }
    std::cout << "sum: " << sum << std::endl;
}

void test_vector_insert()
{
    std::cout << "\n# test_vector_insert\n";

    std::vector<int> z0 {2,23};
    std::cout << "vector<int>{2,23}: " << z0 << std::endl;

    z0.insert(z0.begin(), -2);
    std::cout << ".insert(begin(), -2): " << z0 << std::endl;

    z0.insert(z0.end(), 233);
    std::cout << ".insert(end(), 233): " << z0 << std::endl;

    z0.insert(z0.end(), {2333,2333});
    std::cout << ".insert(end(), {2333,2333}): " << z0 << std::endl;
}

// g++ draft_vector.cpp -std=c++11 -o tbd00.exe
int main(int argc, char const *argv[])
{
    std::cout << "draft_vector.cpp\n";

    test_vector_constructor();
    test_vector_insert();

    return 0;
}
