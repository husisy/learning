# cmake

1. link
   * [offical-guide](https://cmake.org/cmake/help/v3.16/guide/tutorial/index.html)
   * [modern-cmake](https://cliutils.gitlab.io/modern-cmake/)
   * [effective-modern-cmake](https://gist.github.com/mbinna/c61dbb39bca0e4fb7d1f73b0d66a4fd1)
   * [official-site](https://cmake.org)
   * [documentation](https://cmake.org/documentation/)
   * [a-simple-tutorial](https://cmake.org/examples)
   * [cmake-tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/index.html)
   * [CMake入门实战](https://www.hahack.com/codes/cmake/)
   * [vscode-cmake-tools](https://github.com/microsoft/vscode-cmake-tools), [documentation](https://vector-of-bool.github.io/docs/vscode-cmake-tools/index.html)
   * [stackoverflow-QA](https://stackoverflow.com/questions/8304190/cmake-with-include-and-source-paths-basic-setup)
2. build system
   * different computers
   * continuous integration (CI)
   * different OS
   * different compilers
3. 安装
   * `pip install cmake` ，见[pypi](pypi.org)
   * `conda install cmake`，见[anaconda](anaconda.org)
4. 选择compiler
   * linux: `CC=clang CXX=clang++ cmake ..`
   * windows不适用
5. concept
   * compiler: `CC=clang CXX=clang++ cmake ..`
   * generator: `cmake -G "MinGW Makefiles"`
   * regular variable, cache variables
6. 可选参数
   * 在`build`目录下执行`cmake -LH`
   * `-DCMAKE_BUILD_TYPE=`: `Release`, `RelWithDebInfo`, `Debug`
   * `-DCMAKE_INSTALL_PREFIX=`
   * `-DSHARED_LIBS=`
   * `-DBUILD_TESTING=`
7. 关键字
   * `VERSION`
   * `PROJECT_BINARY_DIR`
   * `PROJECT_SOURCE_DIR`
8. 偏见
   * **禁止**向前兼容`cmake_minimum_required(VERSION 3.15)`，没有必要向前兼容，cmake总是可以通过conda/virtualenv创建新环境来安装最新的cmake，[cmake-policy](https://cmake.org/cmake/help/latest/manual/cmake-policies.7.html)
   * `cmake_minimum_required`**必须**放在第一行
   * **禁止**使用全局函数`link_directories() include_libraries()`
   * **禁止**使用`cmake`来构建项目，**必须**在`build/`目录下使用`cmake -S .. -B .`的格式
9. 可执行文件`add_executable(one two.cpp three.h)`
    * 虽然`three.h`会被cmake省略，还是有必要添加于此处以便于IDE分析

workspace

1. `ws00` basic example
2. `ws01` split src include
3. `ws02` add static library, add external static library (windows fail)

## ws-none

测试cmake空项目，用于打印变量操作之类的，see [link](https://github.com/Wigner-GPU-Lab/Teaching/tree/master/CMake/Lesson0_HelloCMake)

```bash
mkdir build
cd build
cmake -S .. -B .
```

## minimum working example ws00

`./draft.cpp`

```cpp
#include <iostream>
int main(){
    std::cout << "hello world from draft00.cpp/main/line4" << std::endl;
}
```

`./CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.15)
project(ws00 VERSION 0.1 DESCRIPTION "project-ws00" LANGUAGES CXX)
add_executable(tbd_run draft.cpp)
```

运行环境：Ubuntu-18.04, gcc-7.4, cmake-3.14；或运行环境：win10, cmake-3.16

```bash
mkdir build
cd build
cmake -S .. -B .
cmake --build .
./tbd_run00 #linux
# ./Debug/tbd_run00.exe #windows
```

linux下对应的编译命令`g++ draft00.cpp -std=c++11 -o tbd_run00.exe`

## ws03

通过`target_link_libraries`来链接静态文件

```bash
mkdir build
cd build
cmake -S .. -B . #use default MY_TAG00=ON
cmake --build .
./tbd_run00
```

linux下对应的编译命令

```bash
mkdir tbd00
g++ -c utils.cpp -o tbd00/utils.o
g++ -c draft00.cpp -o tbd00/draft00.o
g++ tbd00/utils.o tbd00/draft00.o -o tbd00/tbd_run00
./tbd00/tbd_run00
```

## ws06

1. 通过`configure_file`可以从CMakeLists传递字面值至hpp文件
2. 通过`option`可以从CMakeLists传递bool型变量至hpp文件，`option`**必须**放在`configure_file`之前，否则`option`不会发挥作用
3. 通过`-DMY_TAG00=ON/OFF`从命令行传递参数

```bash
mkdir build
cd build
cmake -S .. -B . #use default MY_TAG00=ON
cmake --build .
./tbd_run00

cmake -S .. -B . -DMY_TAG00=OFF #set MY_TAG00=OFF
cmake --build .
./tbd_run
#here if rerun "cmake -S .. -B ." without specify MY_TAG00 explicitly, then MY_TAG00 will remains OFF
```

linux下对应的编译命令

```bash
#g++ -Dxx=yy #since we use draft00.hpp.in here, it doesn't work
```

## misc

1. `ws00`: minimum working example
2. `ws01`: 多个文件
3. `ws02`：多个文件夹，静态链接库
   * TODO 动态链接编译如何做
4. `ws03`：可选参数
