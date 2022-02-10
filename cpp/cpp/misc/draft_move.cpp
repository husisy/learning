#include <iostream>
#include <string>
#include <vector>

void hf0(std::string &)
{
    std::cout << "hf0(left-value) is called" << std::endl;
}

void hf0(std::string &&)
{
    std::cout << "hf0(right-value) is called" << std::endl;
}

void test_lvalue_rvalue()
{
    std::cout << "\n# test_lvalue_rvalue\n";

    std::string x("233");
    std::cout << "calling hf0(x): ";
    hf0(x);
    std::cout << "calling hf0(x+\"3\"): ";
    hf0(x + "3");
}

void test_move()
{
    std::cout << "\n# test_move\n";
    std::string x("233");
    std::cout << "before moving, x: " << x << std::endl;
    std::string y = std::move(x);
    //std::string&& y = std::move(x); // ERROR usage (no exception)
    std::cout << "after moving, x: " << x << std::endl;
    std::cout << "after moving, y: " << y << std::endl;
}

class Class00 {
public:
    int id;
    Class00(int id_):id(id_) {
        std::cout << "construction(" << id_ << ")" << std::endl;
    }
    Class00(Class00& a):id(a.id + 23) {
        std::cout << "copy(" << a.id << ")" << std::endl;
    }
    Class00(Class00&& a):id(a.id+233) {
        //"delete nullptr" is valid
        std::cout << "move(" << a.id << ")" << std::endl;
    }
    ~Class00(){
        std::cout << "delete(" << id << ")" << std::endl;
    }
};

//use "test" to prevent optimization
Class00 hf1(bool test){
    Class00 x(0), y(1);
    if (test)
        return x;
    else
        return y;
}

//another better example see https://zhuanlan.zhihu.com/p/55229582
void test_move_construction()
{
    std::cout << "\n# test_move_construction\n";
    Class00 z0 = hf1(false);
    std::cout << "z0.id: " << z0.id << std::endl;
}

// g++ draft_move.cpp -std=c++17 -o tbd00.exe
int main(int argc, char *argv[])
{
    std::cout << "# draft_move.cpp" << std::endl;

    test_lvalue_rvalue();
    test_move();
    test_move_construction();

    return 0;
}
