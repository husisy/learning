#include <iostream>
#include <vector>
#include <string>
#include <set>

void test_set_constructor()
{
    // https://en.cppreference.com/w/cpp/container/set/set
    std::cout << "\n# test_set_constructor\n";

    std::set<std::string> z0;
    z0.insert("2");
    z0.insert("23");
    z0.insert("233");
    std::cout << "default constructor: ";
    for (auto &x: z0) std::cout << x << ", ";
    std::cout << std::endl;

    std::set<std::string> z1(z0.find("23"), z0.end());
    std::cout << "iterator constructor: ";
    for (auto &x: z1) std::cout << x << ", ";
    std::cout << std::endl;

    std::set<std::string> z2(z0);
    std::cout << "copy constructor: ";
    for (auto &x: z2) std::cout << x << ", ";
    std::cout << std::endl;

    std::set<std::string> z3(std::move(z0));
    std::cout << "copy constructor: ";
    for (auto &x: z3) std::cout << x << ", ";
    std::cout << std::endl;
    std::cout << "\tthe original set after moving: ";
    for (auto &x: z0) std::cout << x << ", ";
    std::cout << std::endl;

    std::set<std::string> z4 {"2", "23", "233"};
    std::cout << "initializer list constructor: ";
    for (auto &x: z4) std::cout << x << ", ";
    std::cout << std::endl;
}


// g++ draft_set.cpp -std=c++11 -o tbd00.exe
int main(int argc, char *argv[])
{
    std::cout << "# draft_set.cpp" << std::endl;

    test_set_constructor();

    std::cout << std::endl;
    return 0;
}
