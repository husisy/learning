#include <iostream>
#include <cmath>

class CArray{
public:
    double* data;
    size_t size;
    CArray(size_t size_): size(size){
        data = new double[size];
    }
    void print_info(){
        std::cout << "data: ";
        for (size_t x=0; x<size; x++){
            std::cout << data[x] << ",";
        }
        std::cout << std::endl;
    }
    ~CArray(){
        if (data)
        {
            std::cout << "delete CArray data" << std::endl;
            delete[] data;
        }
    }
};

// g++ cdraft00.cpp -std=c++11 -o tbd00.exe
int main()
{
    CArray z0(3);
    std::cout << "log2(16): " << (int) std::rint(std::log2((double) 16)) << std::endl;
    return 0;
}
