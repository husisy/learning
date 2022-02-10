cmake_minimum_required(VERSION 3.10)

project(ws02 VERSION 0.1.2 LANGUAGES CXX)

add_executable(tbd_run draft00.cpp)

add_library(utils SHARED IMPORTED)
set_target_properties(utils PROPERTIES IMPORTED_LOCATION "${PROJECT_SOURCE_DIR}/libutils.a")

# target_link_libraries(tbd_run "${PROJECT_SOURCE_DIR}/libutils.a") #linux
target_link_libraries(tbd_run utils) #win
