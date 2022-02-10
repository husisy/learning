#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <set>
#include <tuple>

template <typename Map>
void print_map(Map &x)
{
    std::cout << "{";
    for (auto &y : x)
    {
        std::cout << y.first << ":" << y.second << ", ";
    }
    std::cout << "}\n";
}

void test_map_constructor()
{
    // https://en.cppreference.com/w/cpp/container/map/map
    std::cout << "\n# test_map_constructor\n";

    std::map<std::string, int> z0;
    z0["a"] = 2;
    z0["ab"] = 23;
    z0["abb"] = 233;
    std::cout << "default constructor: ";
    print_map(z0);

    std::map<std::string, int> z1(z0.find("ab"), z0.end());
    std::cout << "iterator constructor: ";
    print_map(z1);

    std::map<std::string, int> z2(z0);
    std::cout << "copy constructor: ";
    print_map(z2);

    std::map<std::string, int> z3(std::move(z0));
    std::cout << "move constructor: ";
    print_map(z3);
    std::cout << "\t the original map after moving: ";
    print_map(z0);

    std::map<std::string, int> z4{{"a", 2}, {"ab", 23}, {"abb", 233}};
    std::cout << "initializer list constructor: ";
    print_map(z4);
}

class MyClass00
{
};

void test_pointer_as_key()
{
    std::cout << "\n# test_pointer_as_key\n";

    MyClass00 x0, x1, x2;
    std::map<MyClass00 *, std::string> z0;
    z0[&x0] = "0x&";
    z0[&x1] = "1x&";
    z0[&x2] = "2x&";

    //the duty of a function: make sure the return value is valid, NOT check that the received pointer is valid
    std::cout << "dict[address of x0]: " << z0[&x0] << std::endl;
    std::cout << "dict[address of x1]: " << z0[&x1] << std::endl;
    std::cout << "dict[address of x2]: " << z0[&x2] << std::endl;
}

void test_map_iterator()
{
    std::cout << "\n# test_map_iterator\n";
    int x0 = 2, x1 = 23, x2 = 233;
    std::map<int *, int> z0{{&x0, x0}, {&x1, x1}, {&x2, x2}};
    auto tmp0 = z0.find(&x2);
    if (tmp0 != z0.end())
    {
        std::cout << "&x2: " << &x2 << std::endl;
        std::cout << "map.find(&x2)->first: " << tmp0->first << std::endl;
        std::cout << "map.find(&x2)->second: " << tmp0->second << std::endl;
    }
}

void test_tuple_as_key()
{
    std::cout << "\n# test_tuple_as_key\n";
    using tuple_ii = std::tuple<int, int>;

    std::map<tuple_ii, std::string> z0;
    z0[tuple_ii(2, 33)] = "233";
    std::cout << "map.find(tuple(2,33)): " << z0[tuple_ii(2, 33)] << std::endl;
}

// g++ draft_map.cpp -std=c++11 -o tbd00.exe
int main(int argc, char *argv[])
{
    std::cout << "# draft00_vector_set_map.cpp" << std::endl;

    test_map_constructor();
    test_pointer_as_key();
    test_map_iterator();
    test_tuple_as_key();

    std::cout << std::endl;
    return 0;
}
