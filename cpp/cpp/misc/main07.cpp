#include <iostream>
#define LEN 10

int len_foo()
{
    int i = 2;
    return i;
}

constexpr int len_foo_constexpr()
{
    return 5;
}

constexpr int fibonacci(const int n)
{
    return (n == 1 || n == 2) ? 1 : (fibonacci(n - 1) + fibonacci(n - 2));
}

int main(){
    char arr_1[10];
    char arr_2[LEN];

    // int len_3 = 10;
    // char arr_3[len_3];

    // const int len_4 = 10;
    // char arr_4[len_4];

    constexpr int len_5 = 1 + 2 + 3;
    char arr_5[len_5];

    char len_6[fibonacci(10)];

    return 0;
}
