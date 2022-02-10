#include <iostream>
using std::cin, std::cout, std::endl;

class ZC1
{
  private:
    int x, y;

  public:
    ZC1(int x_ = 0, int y_ = 0) : x(x_ + 3), y(y_ + 3) { cout << "call ZC1(int: " << x_ << "," << y_ << "): (" << x << "," << y << ")" << endl; }
    ZC1(const ZC1 &z1) : x(z1.x + 5), y(z1.y + 5) { cout << "call ZC1(ZC1 &:" << z1.x << "," << z1.y << "): (" << x << "," << y << ")" << endl; }
    ZC1 &operator=(const ZC1 &z1);
    int sum(int z);
    void hf1(ZC1 &z1) {cout << "call ZC1::hf1(ZC1&: " << z1.x << "," << z1.y << ")" << endl;}
    void hf2(ZC1 z1) {cout << "call ZC1::hf2(ZC1: " << z1.x << "," << z1.y << ")" << endl;}
    ZC1 hf3();
};
ZC1 &ZC1::operator=(const ZC1 &z1)
{
    x = z1.x + 7;
    y = z1.y + 7;
    cout << "call &operator=(ZC1 &" << z1.x << "," << z1.y << "): (" << x << "," << y << ")" << endl;
    return *this;
}
int ZC1::sum(int z = 11)
{
    cout << "call ZC1::sum(int " << z << ")" << endl;
    return x + y + z;
}
ZC1 ZC1::hf3()
{
    cout << "call ZC1::hf3()" << endl;
    return *this;
}

int main()
{
    cout << endl
         << "test basic_class" << endl;
    ZC1 z1 = 2;
    cout << endl;

    cout << "sizeof(z1): " << sizeof(z1) << endl;
    cout << "sizeof(int): " << sizeof(3) << endl;
    cout << endl;

    ZC1 *pz1 = &z1;
    cout << "ZC1* -> sum(): " << pz1->sum() << endl;
    cout << "(*ZC1*).sum(): " << (*pz1).sum() << endl;

    ZC1 &rz1 = z1;
    cout << "ZC1& .sum(): " << rz1.sum() << endl;

    cout << "ZC1(3, 5).sum(7): " << ZC1(3, 5).sum(7) << endl;
    cout << endl;

    z1 = 13;
    cout << endl;

    z1.hf1(z1);
    cout << endl;

    z1.hf2(z1);
    cout << endl;

    ZC1 Z2(z1.hf3());
    cout << endl;

    ZC1 z3[3] = {3, 5, ZC1(7, 11)};
    cout << endl;

    ZC1 *z4[2] = {new ZC1(), new ZC1(3, 5)};
    delete z4[0];
    delete z4[1];
    ZC1 *z5 = new ZC1[2];
    delete z5;
    cout << endl;

    return 0;
}