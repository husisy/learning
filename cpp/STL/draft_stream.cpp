#include <iostream>
#include <string>
#include <fstream>

void test_endless_cin()
{
    std::cout << "\n# test_endless_cin\n";
    std::cout << "endless std::cin>>int, type ctrl+z(win) ctrl+d(*nix) to stop: ";

    int x;
    while (std::cin >> x) //ctrl+z to break
    {
        std::cout << "cout: " << x + 1 << std::endl
                  << "endless std::cin, type ctrl+z(win) ctrl+d(*nix) to stop: ";
    }
}

void test_cout_char_array()
{
    std::cout << "\n# test_cout_char_array\n";
    char z0[4] = {'2', '3', '3'};
    std::cout << "[4]{'2', '3', '3'}: " << z0 << std::endl;
    // char z1[3] = {'2', '3', '3'}; //cout error, char array end without '\0'
    // std::cout << "[3]{'2', '3', '3'}: " << z1 << std::endl;
    char z2[4] = {'2', '\0', '3'};
    std::cout << "[4]{'2', '/0', '3'}: " << z2 << std::endl;
    std::cout << "test char with /0: "
              << "2" << '\0' << "33" << std::endl;
    std::cout << "test string with /0:"
              << "a\0bb"
              << "2"
              << "\0"
              << "33" << std::endl;
}

void test_fstream()
{
    std::cout << "\n# test_fstream()\n";
    std::string filepath("tbd00/tbd00.txt");
    {
        std::ofstream fid(filepath, std::ios::binary);
        fid << 2.33 << "\n";
        fid.close();
    }
    double z0;
    std::ifstream fid(filepath, std::ios::binary);
    fid >> z0;
    std::cout << "reading from " << filepath << std::endl;
    std::cout << "double z0=" << z0 << std::endl;
}

// g++ draft_stream.cpp -std=c++11 -o tbd00.exe
int main(int argc, char *argv[])
{
    std::cout << "# draft_stream.cpp" << std::endl;

    test_endless_cin();
    test_cout_char_array();
    test_fstream();

    std::cout << std::endl;
    return 0;
}
