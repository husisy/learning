cmake_minimum_required(VERSION 3.12)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

project(draft LANGUAGES CXX CUDA)
# set(CMAKE_CUDA_ARCHITECTURES 52;70;75;86)

# kernel in kernel
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_executable(draft00.exe draft00.cu)
add_executable(draft01.exe draft01.cu)
add_executable(draft02.exe draft02.cu)
add_executable(draft03.exe draft03.cu)

target_include_directories(draft02.exe PUBLIC include)
target_include_directories(draft03.exe PUBLIC include)

target_compile_options(draft03.exe PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
