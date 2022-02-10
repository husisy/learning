#include <iostream>

class Demo
{
  public:
    int id;
    Demo(const Demo &z1)
    {
        std::cout << "step in copy constructor" << std::endl;
        id = z1.id;
        std::cout << "step out copy constructor" << std::endl;
    }
    Demo() {}
    Demo(int id_)
    {
        id = id_;
        std::cout << "id = " << id << " constructed" << std::endl;
    }
    ~Demo()
    {
        std::cout << "id = " << id << " destructed" << std::endl;
    }
};

Demo z1(1);
void hf1()
{
    std::cout << "step in hf1" << std::endl;
    static Demo z2(2);
    Demo z3(3);
    std::cout << "step out hf1" << std::endl;
}

// g++ main03.cpp -std=c++11 -o tbd00.exe
int main()
{
    std::cout << "step in main" << std::endl;
    Demo z4(4);
    z4 = 6;
    {
        std::cout << "step in sub-region" << std::endl;
        Demo z5(5);
        std::cout << "step out sub-region" << std::endl;
    }
    hf1();
    std::cout << "step out main" << std::endl;
}
