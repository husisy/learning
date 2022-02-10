#include <iostream>

using std::cin, std::cout, std::endl;

class ZC1
{
  public:
    int n;
    ZC1(int n_) : n(n_ + 1) { cout << "ZC1(int " << n_ << "): " << n << endl; }
};

class ZC2 : public ZC1
{
  public:
    ZC2(int n_) : ZC1(n_ + 1) { cout << "ZC2(INT " << n_ << ") " << ZC1::n << endl; }
};

void basic_extend()
{
    ZC2 z1(2);
}

int main()
{
    basic_extend();
    return 0;
}