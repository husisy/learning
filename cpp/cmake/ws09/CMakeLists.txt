cmake_minimum_required (VERSION 3.10)

project (Demo)

aux_source_directory(. DIR_SRCS)

add_subdirectory(math)

add_executable(Demo.exe ${DIR_SRCS})

target_link_libraries(Demo.exe MathFunctions)
