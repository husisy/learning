#include <string>
#include <sstream>
#include <iostream>

using std::string, std::ostringstream, std::cin, std::cout, std::endl;

class MyComplex
{
  private:
    double real = 0;
    double imag = 0;

  public:
    MyComplex(double real_ = 0, double imag_ = 0) : real(real_), imag(imag_) {}
    MyComplex(const MyComplex &z1)
    {
        real = z1.real;
        imag = z1.imag;
    }
    string to_string()
    {
        ostringstream buffer;
        buffer << real << " + " << imag << "i";
        return string("MyComplex(") + string(buffer.str()) + string(")");
    }
    MyComplex operator+(const MyComplex &z1) { return MyComplex(real + z1.real, imag + z1.imag); }
    MyComplex operator-(const MyComplex &z1) { return MyComplex(real - z1.real, imag - z1.imag); }
    MyComplex operator*(const MyComplex &z1)
    {
        double new_real = real * z1.real - imag * z1.imag;
        double new_imag = imag * z1.real + real * z1.imag;
        return MyComplex(new_real, new_imag);
    }
};

void print_string()
{
    cout << endl
         << "test print_string" << endl;
    MyComplex z1(3, 5), z2(5, 7), z3;
    cout << z1.to_string() << endl;
    cout << z2.to_string() << endl;
    cout << z3.to_string() << endl;
}

void operator_overload()
{
    cout << endl
         << "test operator_overload" << endl;
    MyComplex z1(3, 5), z2(5, 7), z3;
    cout << z1.to_string() << " + " << z2.to_string() << " = " << (z1 + z2).to_string() << endl;
    cout << z1.to_string() << " - " << z2.to_string() << " = " << (z1 - z2).to_string() << endl;
    cout << z1.to_string() << " * " << z2.to_string() << " = " << (z1 * z2).to_string() << endl;
}

int main()
{
    print_string();
    operator_overload();
    return 0;
}
