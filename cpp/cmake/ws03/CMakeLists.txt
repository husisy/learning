cmake_minimum_required(VERSION 3.15)
project(ws01 VERSION 0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)


add_library(utils utils.cpp)

add_executable(tbd_run00 draft00.cpp)

target_link_libraries(tbd_run00 PUBLIC utils)
