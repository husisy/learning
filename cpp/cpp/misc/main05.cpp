#include <iostream>
#include <vector>

using std::cout;
using std::endl;
using std::vector;

int main()
{
    auto hf0 = [](int y) { return y + 233; };
    cout << hf0(233) << endl;

    int x0 = 233;
    auto hf1 = [x0](int y) {return y+x0;};
    x0 = 234;
    cout << hf1(233) << endl;

    int x1 = 233;
    auto hf2 = [&x1](int y) {return y+x1;};
    x1 = 234;
    cout << hf2(233) << endl;

    auto hf3 = [](auto y0, auto y1) {return y0+y1;};
    cout << hf3(233, 233) << endl;
    return 0;
}
