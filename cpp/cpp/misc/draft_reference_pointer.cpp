#include <iostream>

void test_variable_reference()
{
    std::cout << "\n# test_variable_reference\n";
    int n = 3;
    int &r = n;
    r = 5;
    std::cout << "set r=5:: n: " << n << "; r: " << r << std::endl;
    n = 7;
    std::cout << "set n=7:: n: " << n << "; r: " << r << std::endl;
}

void hf1_swap_variable(int &a, int &b)
{
    int tmp1;
    tmp1 = a;
    a = b;
    b = tmp1;
}
void test_swap_variable()
{
    std::cout << "\n# test_swap_variable\n";
    int a = 3;
    int b = 5;
    std::cout << "before swap:: a: " << a << "; b: " << b << std::endl;
    hf1_swap_variable(a, b);
    std::cout << "after swap::  a: " << a << "; b: " << b << std::endl;
}

void test_const_pointer()
{
    std::cout << "\n# test_const_pointer\n";
    int a = 3;
    int b = 5;
    const int *p = &a;
    std::cout << "*p=&a:: " << *p << std::endl;
    p = &b;
    std::cout << "*p=&b:: " << *p << std::endl;
}

void test_new_variable()
{
    std::cout << "\n# test_new_variable\n";
    int *z1 = new int(2);
    std::cout << "new int(2): " << *z1 << std::endl;
    delete z1;

    z1 = new int[2];
    z1[0] = 23;
    z1[1] = 233;
    std::cout << "sizeof(int*): " << sizeof(z1) << std::endl;
    std::cout << "sizeof(z1[0]): " << sizeof(z1[0]) << std::endl; //shenmegui, why fail on std::begin(z1)
    std::cout << "new int[2]: "
         << z1[0] << ", "
         << z1[1] << ", "
         << std::endl;
    delete[] z1;
}

std::string print_int_array(int *p)
{
    int num = sizeof(p) / sizeof(p[0]);//p needs to contains at least one element, not NULL, not nullptr
    std::string ret;
    for (int i=0; i<num; i++)
    {
        ret += std::to_string(p[i]) + ",";
    }
    return ret;
}

void test_poa_or_aop()
{
    std::cout << "\n# test_poa_or_aop, pointer of array, array of pointer\n";
    int *aop[2]; //same as "int* aop[2];"
    int (*poa)[2];

    aop[0] = new int[1] {2};
    aop[1] = new int[3] {2,23,233};
    std::cout << "aop[0]: " << print_int_array(aop[0]) << std::endl;
    std::cout << "aop[1]: " << print_int_array(aop[1]) << std::endl;
    delete[] aop[0];
    delete[] aop[1];

    poa = new int[3][2] {{2,23},{23,233},{233,2333}};
    std::cout << "poa[0]: " << print_int_array(poa[0]) << std::endl;
    std::cout << "poa[1]: " << print_int_array(poa[1]) << std::endl;
    std::cout << "poa[2]: " << print_int_array(poa[2]) << std::endl;
    delete[] poa;
}

// g++ draft_reference_pointer.cpp -std=c++17 -o tbd00.exe
int main(int argc, char *argv[])
{
    std::cout << "# draft_reference_pointer.cpp" << std::endl;

    test_variable_reference();
    test_swap_variable();
    test_const_pointer();
    test_new_variable();
    test_poa_or_aop();

    std::cout << std::endl;
    return 0;
}
