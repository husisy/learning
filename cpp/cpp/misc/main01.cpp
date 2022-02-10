#include <iostream>

using std::cout, std::cin, std::endl;

const char *hf1_function_pointer(const char *s)
{
    cout << s << endl;
    return "world";
}
void function_pointer()
{
    cout << endl
         << "test function_pointer" << endl;
    const char *(*hf1)(const char *);
    hf1 = hf1_function_pointer;
    cout << hf1("hello") << endl;
}

void variable_reference()
{
    cout << endl
         << "test variable_reference" << endl;
    int n = 3;
    int &r = n;
    r = 5;
    cout << "set r=5:: n: " << n << "; r: " << r << endl;
    n = 7;
    cout << "set n=7:: n: " << n << "; r: " << r << endl;
}

void hf1_swap_variable(int &a, int &b)
{
    int tmp1;
    tmp1 = a;
    a = b;
    b = tmp1;
}
void swap_variable()
{
    cout << endl
         << "test swap_variable" << endl;
    int a = 3;
    int b = 5;
    cout << "before swap:: a: " << a << "; b: " << b << endl;
    hf1_swap_variable(a, b);
    cout << "after swap::  a: " << a << "; b: " << b << endl;
}

void const_pointer()
{
    cout << endl
         << "test const_pointer" << endl;
    int a = 3;
    int b = 5;
    const int *p = &a;
    cout << "*p=&a:: " << *p << endl;
    p = &b;
    cout << "*p=&b:: " << *p << endl;
}

void constant_variable()
{
    const int MAX_VAL = 32;
    const double PI = 3.14;
    const char *SCHOOL_NAME = "Peking University";
}

void new_variable()
{
    cout << endl
         << "test new_variable" << endl;
    int *z1 = new int;
    *z1 = 3;
    cout << "*z1=3: " << *z1 << endl;
    delete z1;

    z1 = new int[2];
    z1[0] = 1;
    cout << "new int[4]: "
         << z1[0] << " "
         << z1[1] << " "
         << endl;
    delete[] z1;
}

int hf1_overload(int a, int b)
{
    cout << "hf1_overload(int, int): ";
    if (a > b)
        return a;
    return b;
}
double hf1_overload(double a, double b)
{
    cout << "hf1_overload(double, double): ";
    if (a > b)
        return a;
    return b;
}
void function_overload()
{
    cout << endl
         << "test function_overload" << endl;
    cout << hf1_overload(3, 4) << endl;
    cout << hf1_overload(3.0, 4.0) << endl;
}


int hf1_function_default_argument(int x1, int x2=0, int x3=0)
{
    return x1 + x2 + x3;
}
void function_default_argument()
{
    cout << endl
         << "test function_default_argument" << endl;
    cout << "hf1(1):     " << hf1_function_default_argument(1) << endl;
    cout << "hf1(1,2):   " << hf1_function_default_argument(1,2) << endl;
    cout << "hf1(1,2,3): " << hf1_function_default_argument(1,2,3) << endl;
}

int main(int argc, char *argv[])
{
    cout << "test argc / argv" << endl;
    cout << "argc: " << argc << endl;
    for (int i = 0; i < argc; i++)
        cout << "argv: " << argv[i] << endl;

    function_pointer();

    variable_reference();

    swap_variable();

    const_pointer();

    new_variable();

    function_overload();

    function_default_argument();
    return 0;
}
