cmake_minimum_required(VERSION 3.15)
project(ws00 VERSION 0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

message("[myinfo] CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}")
message("[myinfo] CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

add_executable(tbd_run draft00.cpp)
