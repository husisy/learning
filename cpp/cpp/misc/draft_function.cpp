#include <iostream>
#include <initializer_list>

int hf0()
{
    static int x = 0;
    x++;
    return x;
}

void test_static()
{
    std::cout << "\n# test_static\n";

    std::cout << "hf0(): " << hf0() << std::endl;
    std::cout << "hf0(): " << hf0() << std::endl;
    std::cout << "hf0(): " << hf0() << std::endl;
}

const char *hf0(const char *s)
{
    std::cout << s << std::endl;
    return "world";
}
void test_function_pointer()
{
    std::cout << "\n# test_function_pointer\n";
    const char *(*hf1)(const char *);
    hf1 = hf0;
    std::cout << hf1("hello") << std::endl;
}

template <class T>
void hf1(std::initializer_list<T> args)
{
    std::cout << "receiving variable arguments (same type): hf1(";
    for (auto &x : args)
    {
        std::cout << x << ",";
    }
    std::cout << ")" << std::endl;
}

template <typename T>
void func(T t)
{
    std::cout << t << std::endl;
}

// template <typename... Args>
// void hf2(Args... args)
// {
//     std::cout << "receiving variable arguments (different type): hf1(";
//     for (auto &x: args...)
//     {
//         std::cout << x << ",";
//     }
//     std::cout << ")" << std::endl;
// }

void test_variable_argument()
{
    std::cout << "\n# test_variable_argument\n";
    hf1({"2", "23", "233", "2333"});
    // hf2(2, "23", 233, "233"); //fail, see https://stackoverflow.com/a/16338804
}

std::string hf2(int a, int b)
{
    return "calling hf2(int, int)";
}
std::string hf2(double a, double b)
{
    return "calling hf2(double, double)";
}
void test_function_overload()
{
    std::cout << "\n# test_function_overload\n";
    std::cout << "hf2(23,233): " << hf2(23, 233) << std::endl;
    std::cout << "hf2(23.0,233.0): " << hf2(23.0, 233.0) << std::endl;
}

std::string hf3(int x1, int x2=0)
{
    return std::string("hf3(") + std::to_string(x1) + "," + std::to_string(x2) + ")";
}
void test_function_default_argument()
{
    std::cout << "\n# test_function_default_argument\n";
    std::cout << "hf3(233): " << hf3(233) << std::endl;
    std::cout << "hf3(233, 233): " << hf3(233,233) << std::endl;
}

// g++ draft_function.cpp -std=c++17 -o tbd00.exe
int main(int argc, char *argv[])
{
    std::cout << "# draft_function.cpp" << std::endl;

    test_static();
    test_function_pointer();
    test_variable_argument();
    test_function_overload();
    test_function_default_argument();

    std::cout << std::endl;
    return 0;
}
