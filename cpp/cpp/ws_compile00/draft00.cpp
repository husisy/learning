#include <iostream>

#include "utils.h"


// g++ -c draft00.cpp -o draft00.o
// g++ -c utils.cpp -o utils.o
// g++ draft00.o utils.o -o tbd00.exe

// g++ draft00.cpp utils.cpp -o tbd00.exe
int main()
{
    hf0();
    std::cout << "[draft00.main()] hello world" << std::endl;
    return 0;
}
