#include <iostream>
#include <vector>
#include <algorithm>

using std::cout;
using std::endl;
using std::vector;

// https://changkun.de/modern-cpp/book/02-usability/index.html#decltype
void main00_decltype()
{
    cout << endl
         << "# main00_decltype" << endl;
    double x0 = 1;
    int x1 = 2;
    decltype(x0 + x1) x2;
    cout << "do NOT depend on typeid() which is not c++ stardard" << endl;
    cout << "double type is: " << typeid(x0).name() << endl;
    cout << "int type is: " << typeid(x1).name() << endl;
    cout << "int + double type is: " << typeid(x2).name() << endl;

    cout << endl;
    cout << "std::is_same<decltype(x0), double>::value: " << std::is_same<decltype(x0), double>::value << endl;
    cout << "std::is_same<decltype(x0), decltype(x1)>::value: " << std::is_same<decltype(x0), decltype(x1)>::value << endl;
}

// https://changkun.de/modern-cpp/book/02-usability/index.html#%E5%B0%BE%E8%BF%94%E5%9B%9E%E7%B1%BB%E5%9E%8B%E6%8E%A8%E5%AF%BC
template <typename T0, typename T1>
auto main01_hf0(T0 x, T1 y)
{
    return x + y;
}
void main01_typeDerivation()
{
    cout << endl
         << "# main01_typeDerivation" << endl;

    auto x0 = main01_hf0(233, 2.33);
    if (std::is_same<decltype(x0), double>::value)
    {
        cout << "auto main01_hf0(int, double) return  double value" << endl;
    }
}

// https://changkun.de/modern-cpp/book/02-usability/index.html#%E5%8C%BA%E9%97%B4-for-%E8%BF%AD%E4%BB%A3
void main02_foreach()
{
    cout << endl
         << "# main02_foreach" << endl;

    vector<int> z0{1, 2, 3, 4};

    if (auto itr0 = std::find(z0.begin(), z0.end(), 3); itr0 != z0.end())
    {
        *itr0 = 4;
    }
    for (auto x : z0)
    {
        cout << x << endl;
    }
    for (auto &x : z0)
    {
        x += 1;
    }
    for (auto x : z0)
    {
        cout << x << endl;
    }
}

int main()
{
    main00_decltype();
    main01_typeDerivation();
    main02_foreach();
    return 0;
}
