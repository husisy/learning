cmake_minimum_required(VERSION 3.10)

project(ws02 VERSION 0.1.2 LANGUAGES CXX)

add_library(utils utils.cpp) #generate libutils.a

add_executable(tbd_run draft00.cpp)

target_link_libraries(tbd_run utils)
