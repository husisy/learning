#include <iostream>
using std::cin, std::cout, std::endl;

class CS0
{
  private:
    int x, y;

  public:
    CS0(int x_ = 0, int y_ = 0) : x(x_ + 3), y(y_ + 3) { cout << "call CS0(int: " << x_ << "," << y_ << "): (" << x << "," << y << ")" << endl; }
    CS0(const CS0 &z1) : x(z1.x + 5), y(z1.y + 5) { cout << "call CS0(CS0 &:" << z1.x << "," << z1.y << "): (" << x << "," << y << ")" << endl; }
    CS0 &operator=(const CS0 &z1);
    int sum(int z);
    void hf1(CS0 &z1) {cout << "call CS0::hf1(CS0&: " << z1.x << "," << z1.y << ")" << endl;}
    void hf2(CS0 z1) {cout << "call CS0::hf2(CS0: " << z1.x << "," << z1.y << ")" << endl;}
    CS0 hf3();
};
CS0 &CS0::operator=(const CS0 &z1)
{
    x = z1.x + 7;
    y = z1.y + 7;
    cout << "call &operator=(CS0 &" << z1.x << "," << z1.y << "): (" << x << "," << y << ")" << endl;
    return *this;
}
int CS0::sum(int z = 11)
{
    cout << "call CS0::sum(int " << z << ")" << endl;
    return x + y + z;
}
CS0 CS0::hf3()
{
    cout << "call CS0::hf3()" << endl;
    return *this;
}

// g++ draft_class.cpp -std=c++17 -o tbd00.exe
int main(int argc, char *argv[])
{
    cout << endl
         << "test basic_class" << endl;
    CS0 z1 = 2;
    cout << endl;

    cout << "sizeof(z1): " << sizeof(z1) << endl;
    cout << "sizeof(int): " << sizeof(3) << endl;
    cout << endl;

    CS0 *pz1 = &z1;
    cout << "CS0* -> sum(): " << pz1->sum() << endl;
    cout << "(*CS0*).sum(): " << (*pz1).sum() << endl;

    CS0 &rz1 = z1;
    cout << "CS0& .sum(): " << rz1.sum() << endl;

    cout << "CS0(3, 5).sum(7): " << CS0(3, 5).sum(7) << endl;
    cout << endl;

    z1 = 13;
    cout << endl;

    z1.hf1(z1);
    cout << endl;

    z1.hf2(z1);
    cout << endl;

    CS0 Z2(z1.hf3());
    cout << endl;

    CS0 z3[3] = {3, 5, CS0(7, 11)};
    cout << endl;

    CS0 *z4[2] = {new CS0(), new CS0(3, 5)};
    // std::vector<CS0> z0 {CS0(), CS0(3,5)};
    delete z4[0];
    delete z4[1];
    CS0 *z5 = new CS0[2];
    delete z5;
    cout << endl;

    return 0;
}