#include <iostream>

class CS1
{
    public:
    int id;
    CS1(int id_): id(id_){
        std::cout << "calling CS1(" << id << ")" << std::endl;
    }
    ~CS1(){
        std::cout << "calling ~CS1(" << id << ")" << std::endl;
    }
};

void test_class_delelte()
{
    std::cout << "\n# test_class_delete.cpp\n";

    CS1 *z0 = new CS1(0);
    CS1 z1 = CS1(1);
}


// g++ draft00.cpp -std=c++11 -o tbd00.exe
// &"C:/Program Files (x86)/Microsoft Visual Studio 14.0/VC/bin/x86_amd64/cl.exe" draft00.cpp -std=c++11 -o tbd00.exe
int main(int argc, char *argv[])
{
    std::cout << "# draft00.cpp" << std::endl;

    test_class_delelte();
    return 0;
}
