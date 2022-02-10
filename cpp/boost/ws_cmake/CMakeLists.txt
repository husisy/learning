cmake_minimum_required(VERSION 3.11.0)

cmake_policy(SET CMP0074 NEW)

project(ws00 VERSION 0.1 LANGUAGES CXX)

find_package(Boost REQUIRED)

add_executable(tbd_run draft00.cpp)

target_include_directories(tbd_run PRIVATE "${Boost_INCLUDE_DIRS}")
