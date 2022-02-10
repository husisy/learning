#include <pybind11/pybind11.h>

// #include "utils.cpp"
#include "utils.hpp"
#include "GPUutils.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_cpp, m)
{
    m.doc() = "zctest_pybind11._cpp";

    py::class_<CArray>(m, "CArray", py::buffer_protocol())
        .def(py::init<int>())
        .def(py::init<py::array_t<double>&, bool>(), py::arg("npdata"), py::arg("copy")=true)
        .def_buffer([](CArray &x) -> py::buffer_info {
            return py::buffer_info(
                x.data,                                  /* Pointer to buffer */
                sizeof(double),                          /* Size of one scalar */
                py::format_descriptor<double>::format(), /* Python struct-style format descriptor */
                1,                                       /* Number of dimensions */
                {x.size},                                /* Buffer dimensions */
                {sizeof(double)});                       /* Strides (in bytes) for each index */
        })
        .def("numpy", &CArray::numpy)
        .def("get_info", &CArray::get_info);
    m.def("c_sum", &c_sum);
    m.def("cuda_plus", &cuda_plus);
    m.def("cuda_plus_return_np", &cuda_plus_return_np);
    m.def("demo_vector_add", &demo_vector_add);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
