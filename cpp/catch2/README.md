# catch2

1. 安装见[github](https://github.com/catchorg/Catch2/blob/master/docs/tutorial.md#getting-catch2)
2. 添加全局`CPLUS_INCLUDE_PATH`
   * powershell: `$env:CPLUS_INCLUDE_PATH += ";C:\Users\zchao\Documents\cplusplus\STL"`
   * bash: `export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:/home/zhangc/STL"`
3. `g++ -std=c++11 -o tbd00.exe draft00.cpp`
4. `tbd00.exe --reporter compact --success`

## installation

单文件下载保存为`catch.hpp`，[github-link](https://raw.githubusercontent.com/catchorg/Catch2/master/single_include/catch2/catch.hpp)，完整运行示例：

文件结构

```bash
.
|--catch.hpp
|--test00.cpp
```

`test00.cpp`

```cpp
#define CATCH_CONFIG_MAIN
#include <catch.hpp>
int Factorial(int number)
{
    return number <= 1 ? 1 : Factorial(number - 1) * number;
}
TEST_CASE("Factorials are computed", "[factorial]")
{
    REQUIRE(Factorial(1) == 1);
    REQUIRE(Factorial(2) == 2);
    REQUIRE(Factorial(3) == 6);
    REQUIRE(Factorial(10) == 3628800);
}
```

1. 编译`g++ -std=c++17 -o tbd00.exe test00.cpp`
2. 运行`./tbd00.exe -reporter compact --success`
   * `./tbd00.exe -r compact -s`，详细参数见`./tbd00.exe --help`

---

cmake integration

使用cmake安装Catch2，[link](https://github.com/catchorg/Catch2/blob/master/docs/cmake-integration.md#installing-catch2-from-git-repository)

1. 常见git/cmake命令
   * `git clone https://github.com/catchorg/Catch2.git`
   * `cd Catch2`, `mkdir build`, `cd build`
   * `cmake -S .. -B . -DBUILD_TESTING=OFF`，已测试win（使用`visual studio 16 2019` generator）
   * `cmake --build .`
2. 安装 `cmake --install . --prefix SOMEWHERE`，将此处以及下文中的`SOMEWHERE`替换为合适的路径（个人路径仅供参考）
   * windows-powershell `C:/Users/zchao/Documents/software/Catch2`
   * linux `/home/zchao/software/Catch2`
3. 设置环境变量
   * windows-powershell: `$env:Catch2_ROOT="SOMEWHERE"`
   * linux-bash: `export Catch2_ROOT="SOMEWHERE"`

完整运行示例：

文件结构

```bash
.
|--CMakeLists.txt
|--test_catch_main.cpp
|--test00.cpp
```

`CMakeLists.txt`

```cmake
cmake_minimum_required(VERSION 3.15)
project(w00 VERSION 0.0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
find_package(Catch2 REQUIRED)
add_executable(test00 test_catch_main.cpp test00.cpp)
target_link_libraries(test00 Catch2::Catch2)
include(CTest)
include(Catch)
catch_discover_tests(test00)
```

`test_catch_main.cpp`

```cpp
#define CATCH_CONFIG_MAIN
#include "catch2/catch.hpp"
```

`test00.cpp`

```cpp
#include "catch2/catch.hpp"
int Factorial(int number)
{
    return number <= 1 ? 1 : Factorial(number - 1) * number;
}
TEST_CASE("Factorials are computed", "[factorial]")
{
    REQUIRE(Factorial(1) == 1);
    REQUIRE(Factorial(2) == 2);
    REQUIRE(Factorial(3) == 6);
    REQUIRE(Factorial(10) == 3628800);
}
```

1. 在`./build`目录下执行`cmake -S .. -B .`
2. 在`./build`目录下执行`cmake --build .`
3. 在`./build`目录下执行`ctest`
