#include <iostream>


void hf0()
{
#ifdef MYTAG
    std::cout << "#ifdef MYTAG is True\n";
#else
    std::cout << "#ifdef MYTAG is False\n";
#endif
#ifndef MYTAG
    std::cout << "#ifndef MYTAG is True\n";
#else
    std::cout << "#ifndef MYTAG is False\n";
#endif
}


// g++ draft_compile_tag.cpp -o tbd00.exe
// g++ -DMYTAG draft_compile_tag.cpp -o tbd00.exe
int main(int argc, char *argv[])
{
    hf0();
    return 0;
}
