#include <string>
#include <iostream>
#include <cassert>
#include <complex>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class Pet
{
public:
    std::string name;
    Pet(const std::string &name_) : name(name_) {}
    void set_name(const ::std::string &name_) { name = name_; }
    const std::string &get_name() const { return name; }
    // ~Pet(){};
};

Pet new_pet()
{
    return Pet("new_pet()");
}

void print_dict(py::dict dict)
{
    for (auto item : dict)
        std::cout << "key=" << std::string(py::str(item.first)) << ", "
                  << "value=" << std::string(py::str(item.second)) << std::endl;
}

int add(int i, int j)
{
    return i + j;
}

void utf8_string(const std::string &s)
{
    std::cout << "utf8_string(const std::string&) is called: " << s << std::endl;
}

void utf8_charptr(const char *s)
{
    std::cout << "utf8_charptr(const char *) is called: " << s << std::endl;
}

std::string return_utf8_string()
{
    return "2333 from return_utf8_string()";
}

char *return_utf8_charptr()
{
    return "2333 from return_utf8_charptr()";
}

class CArray
{
public:
    double *data = nullptr;
    size_t size;
    CArray(size_t size_) : size(size_)
    {
        data = new double[size];
        for (size_t x = 0; x < size; x++)
        {
            data[x] = x;
        }
    }
    CArray(py::buffer b, bool copy=false)
    {
        py::buffer_info info = b.request();
        // assert(info.format==py::format_descriptor<double>::format());
        assert(info.ndim==1);
        size = info.shape[0];
        double *ptr = (double *)info.ptr;
        if (copy){
            data = ptr;
        }else
        {
            data = new double[size];
            size_t delta = info.strides[0]/sizeof(double);
            for (size_t ind0=0; ind0<size; ind0++)
            {
                data[ind0] = *ptr;
                ptr += delta;
            }
        }
    }
    std::string get_info()
    {
        if (data)
        {
            std::string ret = "";
            for (size_t x = 0; x < size; x++)
            {
                ret = ret + std::to_string(data[x]) + ",";
            }
            return ret;
        }
        else
        {
            return "";
        }
    }
    ~CArray()
    {
        if (data)
        {
            std::cout << "delete CArray data" << std::endl;
            delete[] data;
        }
    }
};

double c_sum(py::array_t<double> b)
{
    py::buffer_info info = b.request();
    // assert(info.format==py::format_descriptor<double>::format());
    assert(info.ndim==1);
    size_t size = info.shape[0];
    double ret;
    double *ptr = (double *)info.ptr;
    for (size_t x=0; x<size; x++){
        ret += ptr[x];
    }
    return ret;
}

std::complex<double> c_sum_complex(py::array_t<std::complex<double>> b)
{
    py::buffer_info info = b.request();
    // assert(info.format==py::format_descriptor<double>::format());
    assert(info.ndim==1);
    size_t size = info.shape[0];
    std::complex<double> ret;
    std::complex<double> *ptr = (std::complex<double> *)info.ptr;
    for (size_t x=0; x<size; x++){
        ret = ret + ptr[x];
    }
    return ret;
}
