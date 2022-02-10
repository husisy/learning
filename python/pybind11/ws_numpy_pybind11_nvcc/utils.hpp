#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

class CArray
{
public:
    double *data = nullptr;
    int size;
    bool from_numpy=false;
    CArray(int size_);
    CArray(const CArray &old) : data(old.data), size(old.size){};
    CArray(py::array_t<double> &npdata, bool copy = true);
    CArray(CArray &&old) : size(old.size), data(old.data)
    {
        old.data = nullptr;
        std::cout << "calling Carray(&&)\n";
    };
    py::array_t<double> numpy();
    std::string get_info();
    ~CArray();
};

double c_sum(py::array_t<double> &npdata);

CArray cuda_plus(py::array_t<double> &npdata0, py::array_t<double> &npdata1);

py::array_t<double> cuda_plus_return_np(py::array_t<double> &npdata0, py::array_t<double> &npdata1);
