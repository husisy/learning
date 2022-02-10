#include <iostream>
#include <vector>
#include <string>

void test_string_to_int()
{
    std::cout << "\n# test_string_to_int\n";

    std::cout << "atoi(233): " << std::atoi("233") << std::endl;
    std::cout << "atoi(abb): " << std::atoi("abb") << std::endl;
    std::cout << "stoi(233): " << std::stoi("233") << std::endl;
    // std::cout << "stoi(abb): " << std::stoi("abb") << std::endl; //error
}


// g++ draft_string.cpp -std=c++11 -o tbd00.exe
int main(int argc, char *argv[])
{
    std::cout << "draft_string.cpp\n";

    test_string_to_int();

    return 0;
}
