cmake_minimum_required(VERSION 3.15)
project(ws02 VERSION 0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

message("[myinfo] ws02_VERSION_MAJOR: ${ws02_VERSION_MAJOR}")
message("[myinfo] ws02_VERSION_MINOR: ${ws02_VERSION_MINOR}")
message("[myinfo] PROJECT_BINARY_DIR: ${PROJECT_BINARY_DIR}")

option(MY_TAG00 "just my tag" ON)

configure_file(draft00.hpp.in draft00.hpp)
add_executable(tbd_run00 draft00.cpp)
target_include_directories(tbd_run00 PUBLIC "${PROJECT_BINARY_DIR}")
