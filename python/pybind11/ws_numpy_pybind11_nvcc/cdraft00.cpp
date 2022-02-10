#include <iostream>

class MyClass
{
public:
    double* data = nullptr;
    MyClass(double x)
    {
        std::cout << "call MyClass(double)" << std::endl;
        data = new double[100];
        data[0] = x;
    }
    MyClass(const MyClass &old): data(old.data)
    {
        std::cout << "call MyClass(&)" << std::endl;
    }
    MyClass(MyClass &&old): data(old.data){
        old.data = nullptr;
        std::cout << "call MyClass(&&)" << std::endl;
    }
    ~MyClass()
    {
        if (data)
        {
            std::cout << "call ~MyClass()" << std::endl;
            delete[] data;
        }
    }
};

MyClass some_function(bool test)
{
    std::cout << "call some_function()" << std::endl;
    MyClass tmp0(2.33), tmp1(4.66);
    std::cout << "start return MyClass" << std::endl;
    if (test){
        return tmp0;
    }else
    {
        return tmp1;
    }
}

// g++ cdraft00.cpp -std=c++11 -o tbd00.exe
int main()
{
    std::cout << "call main()" << std::endl;
    MyClass tmp0 = some_function(true);
    std::cout << "bad value" << tmp0.data[0] << std::endl;
    std::cout << "end of main()" << std::endl;
    return 0;
}