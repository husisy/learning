#include <string>
#include <iostream>
#include <cassert>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "utils.hpp"
#include "GPUutils.hpp"

namespace py = pybind11;

CArray::CArray(int size_) : size(size_)
{
    data = new double[size];
    for (int x = 0; x < size; x++)
    {
        data[x] = x;
    }
}

CArray::CArray(py::array_t<double> &npdata, bool copy)
{
    py::buffer_info info = npdata.request();
    // assert(info.format==py::format_descriptor<double>::format());
    assert(info.ndim == 1);
    size = info.shape[0];
    double *ptr = (double *)info.ptr;
    if (copy)
    {
        data = new double[size];
        int delta = info.strides[0] / sizeof(double);
        for (int ind0 = 0; ind0 < size; ind0++)
        {
            data[ind0] = *ptr;
            ptr += delta;
        }
    }
    else
    {
        if (info.strides[0] != sizeof(double))
        {
            std::cout << "CArray(copy=False) do NOT support strides!=sizeof(double)" << std::endl;
            assert(false);
        }
        from_numpy = true;
        data = ptr;
    }
}

std::string CArray::get_info()
{
    if (data)
    {
        std::string ret = "";
        for (int x = 0; x < size; x++)
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

py::array_t<double> CArray::numpy()
{
    // see https://stackoverflow.com/a/44682603/7290857
    py::capsule nothing_when_done(data, [](void *f) { std::cout << "call nothing_when_done @" << f << std::endl; });
    return py::array_t<double>(
        {size},             // shape
        {sizeof(double)},   // C-style contiguous strides for double
        data,               // the data pointer
        nothing_when_done); // numpy array references this parent
}

CArray::~CArray()
{
    if (data && (!from_numpy))
    {
        std::cout << "delete CArray data" << std::endl;
        delete[] data;
    }
}

double c_sum(py::array_t<double> &npdata)
{
    py::buffer_info info = npdata.request();
    assert(info.ndim == 1);
    std::size_t size = info.shape[0];
    double ret;
    double *ptr = (double *)info.ptr;
    for (std::size_t x = 0; x < size; x++)
    {
        ret += ptr[x];
    }
    return ret;
}

CArray cuda_plus(py::array_t<double> &npdata0, py::array_t<double> &npdata1)
{
    CArray cnp0(npdata0, false), cnp1(npdata1, false);
    assert(cnp0.size == cnp1.size);
    int size = cnp0.size;
    CArray ret(size);
    cuda_vector_add(cnp0.data, cnp1.data, size, ret.data);
    return ret;
}

py::array_t<double> cuda_plus_return_np(py::array_t<double> &npdata0, py::array_t<double> &npdata1)
{
    CArray cnp0(npdata0, false), cnp1(npdata1, false);
    assert(cnp0.size == cnp1.size);
    int size = cnp0.size;
    double *data = new double[size];
    cuda_vector_add(cnp0.data, cnp1.data, size, data);

    // see https://stackoverflow.com/a/44682603/7290857
    py::capsule free_when_done(data, [](void *f) {
        double *foo = reinterpret_cast<double *>(f);
        std::cout << "freeing memory @ " << f << "\n";
        delete[] foo;
    });
    return py::array_t<double>(
        {size},           // shape
        {sizeof(double)}, // C-style contiguous strides for double
        data,             // the data pointer
        free_when_done);  // numpy array references this parent
}