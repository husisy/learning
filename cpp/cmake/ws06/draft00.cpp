#include <iostream>
#include <string>

#include "draft00.hpp"

int main(int argc, char* argv[])
{
    std::cout << "hello world from draft00.cpp/main/line4" << std::endl;
    std::cout << "args from CmakeLists.txt: " << draft00_ARGS_FROM_CMakeLists << std::endl;

#ifdef MY_TAG00
    std::cout << "MY_TAG00 from CMakeLists is ON" << std::endl;
#else
    std::cout << "MY_TAG00 from CMakeLists is OFF" << std::endl;
#endif
    return 0;
}
