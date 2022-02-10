# example from https://gitlab.com/CLIUtils/modern-cmake/tree/master/examples/simple-project
cmake_minimum_required(VERSION 3.15)

project(ws10 VERSION 0.1 LANGUAGES CXX)

add_library(utils utils.cpp utils.hpp)

add_executable(tbd_run draft00.cpp)

target_link_libraries(tbd_run PRIVATE utils)

enable_testing()
add_test(NAME tbd_run COMMAND tbd_run)
