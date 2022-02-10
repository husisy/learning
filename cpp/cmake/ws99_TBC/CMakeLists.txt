cmake_minimum_required(VERSION 3.10)

project(ws00 VERSION 0.1 LANGUAGES CXX)

message("[my-info] CMAKE_C_COMPILER: ${CMAKE_C_COMPILER}")
message("[my-info] CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")

add_executable(tbd_run draft00.cpp)


set(variable00 "233")
message("[my-info] variable00 in top dir before calling add-subdirectory(): ${variable00}")
add_subdirectory(tools)
message("[my-info] variable00 in top dir after calling add-subdirectory(): ${variable00}")

# test variable
message("[my-info] regular_variable00: ${regular_variable00}")
set(regular_variable00 "233")
message("[my-info] regular_variable00: ${regular_variable00}")

message("[my-info] cache_variable00: ${cache_variable00}")
set(cache_variable00 "233" CACHE STRING "my-test")
message("[my-info] cache_variable00: ${cache_variable00}")
