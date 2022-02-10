#include <iostream>
#include <cstring>

using std::cin, std::cout, std::endl;
using std::memcpy;

class MyArray
{
  private:
    int *ptr;
    int size;

  public:
    MyArray() : ptr(NULL), size(0){};
    MyArray(const MyArray &z1)
    {
        size = z1.length();
        ptr = new int[z1.size];
        memcpy(ptr, z1.ptr, sizeof(int)*z1.size);
    }
    ~MyArray()
    {
        if (ptr)
            delete[] ptr;
    };
    int length() const { return size; };
    int &operator[](int index) { return ptr[index]; };
    MyArray &operator=(MyArray &z1)
    {
        if (z1.ptr != ptr)
        {
            if (ptr)
                delete[] ptr;
            size = z1.length();
            ptr = new int[z1.size];
            for (int i = 0; i < size; i++)
            {
                ptr[i] = z1[i];
            }
        }
        return *this;
    }
    void print_array()
    {
        if (ptr)
        {
            for (int i = 0; i < size; i++)
            {
                cout << ptr[i] << ", ";
            }
            cout << endl;
        }
    }
    static MyArray range(int num1) { return range(0, num1); }
    static MyArray range(int num1, int num2)
    {
        MyArray ret;
        ret.size = num2 - num1;
        ret.ptr = new int[ret.size];
        for (int i = 0; i < ret.size; i++)
        {
            ret.ptr[i] = i + num1;
        }
        return ret;
    }
};

int main()
{
    MyArray z1(MyArray::range(5)), z2;
    z1.print_array();
    z1[2] = 233;
    z1.print_array();
    z2 = z1;
    z2[0] = 2333;
    z2.print_array();
    z1.print_array();

    return 0;
}