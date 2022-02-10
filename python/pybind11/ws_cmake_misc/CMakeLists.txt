cmake_minimum_required(VERSION 3.15)
project(cmake_example)

find_package(pybind11 REQUIRED)
pybind11_add_module(_cpp main.cpp)
