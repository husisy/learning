#include <iostream>
#include <cstring>

using std::strlen, std::strcpy, std::cin, std::cout, std::endl;

// no two MyString objects point to same "str"
// String.str could not be NULL
class MyString
{
  private:
    char *str;

  public:
    MyString(const char *s = "") : str(new char[strlen(s)+1]) { strcpy(str, s); }
    MyString(const MyString &s) {MyString(s.str);}
    ~MyString(){delete[] str;}
    const char *to_str() { return str; }
    MyString &operator=(const char *s)
    {
        cout << "call &operator=(const char *s)" << endl;
        if (s != str)
        {
            delete[] str;
            str = new char[strlen(s) + 1];
            strcpy(str, s);
        }
        return *this;
    }
    MyString &operator=(const MyString &z1)
    {
        cout << "call &operator=(const MyString &)" << endl;
        return operator=(z1.str);
    }
};

void test_my_string()
{
    cout << endl
              << "test_my_string" << endl;
    MyString z1("world"), z2, z3;
    z2 = "hello ";
    z3 = z1;
    cout << z2.to_str() << z3.to_str() << endl;
}

void test_cstring()
{
    cout << endl
              << "test_ctring" << endl;
    char *p = new char[3];
    p[0] = '0';
    cout << "new char[3]" << endl;
    p[2] = '\0';
    cout << "p[2]=end strlen: " << strlen(p) << endl;
    p[1] = '\0';
    cout << "p[1]=end strlen: " << strlen(p) << endl;
    p[0] = '\0';
    cout << "p[0]=end strlen: " << strlen(p) << endl;
    delete[] p;
}

int main()
{
    test_cstring();

    test_my_string();
    return 0;
}