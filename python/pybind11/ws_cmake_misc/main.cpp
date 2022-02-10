#include <pybind11/pybind11.h>

#include "utils.cpp"

namespace py = pybind11;

PYBIND11_MODULE(_cpp, m)
{
    m.doc() = "zctest_pybind11._cpp";

    m.def("add", &add, R"pbdoc(
        Add two numbers

        Some other explanation about the add function.
    )pbdoc");

    auto tmp0 = [](const Pet &a) { return "<example.Pet named '" + a.name + "'>"; };
    py::class_<Pet>(m, "Pet")
        .def(py::init<const std::string &>())
        .def("set_name", &Pet::set_name)
        .def("get_name", &Pet::get_name)
        .def_readwrite("name", &Pet::name)
        .def("__repr__", tmp0)
        .def("__str__", tmp0);
    m.def("new_pet", &new_pet, "test return Pet");
    m.def("print_dict", &print_dict);

    m.def("utf8_string", &utf8_string);
    m.def("utf8_charptr", &utf8_charptr);
    m.def("return_utf8_string", &return_utf8_string);
    m.def("return_utf8_charptr", &return_utf8_charptr);

    py::class_<CArray>(m, "CArray", py::buffer_protocol())
        .def(py::init<size_t>())
        .def(py::init<py::buffer, bool>(), py::arg("npdata"), py::arg("copy")=false)
        .def_buffer([](CArray &x) -> py::buffer_info {
            return py::buffer_info(
                x.data,                                  /* Pointer to buffer */
                sizeof(double),                          /* Size of one scalar */
                py::format_descriptor<double>::format(), /* Python struct-style format descriptor */
                1,                                       /* Number of dimensions */
                {x.size},                                /* Buffer dimensions */
                {sizeof(double)});                       /* Strides (in bytes) for each index */
        })
        .def("get_info", &CArray::get_info);
    m.def("c_sum", &c_sum);
    m.def("c_sum_complex", &c_sum_complex);

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif
}
