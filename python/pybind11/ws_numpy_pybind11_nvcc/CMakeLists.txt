cmake_minimum_required(VERSION 3.15)
project(cmake_example LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(pybind11 REQUIRED)
set(PYBIND11_CPP_STANDARD -std=c++11)

find_package(CUDA REQUIRED)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED True)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_library(GPUutils STATIC GPUutils.cu)
set_target_properties(GPUutils PROPERTIES POSITION_INDEPENDENT_CODE ON)

pybind11_add_module(_cpp wrapper.cpp utils.cpp)
target_link_libraries(_cpp PRIVATE GPUutils)
