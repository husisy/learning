# thrust

1. link
   * [github/nvidia/thrust](https://github.com/NVIDIA/thrust)
   * [github/nvidia/cub](https://github.com/NVIDIA/cub)
2. installation
   * `sudo apt install libthrust-dev`
   * cmake installation see below (recommand)

cmake installation (see `ws00/`)

```bash
git clone git@github.com:NVIDIA/thrust.git
git clone git@github.com:NVIDIA/cub.git
export Thrust_DIR="/path/to/thrust" #replace it with the above folder
export CUB_DIR="/path/to/cub"
```

`CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.15)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
project(hellocuda LANGUAGES CXX CUDA)
find_package(Thrust REQUIRED CONFIG)
# find_package(CUB REQUIRED)
thrust_create_target(Thrust)

add_executable(draft00.exe draft00.cu)
target_link_libraries(draft00.exe Thrust)
```
