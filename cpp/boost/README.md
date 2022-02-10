# boost

## installation

header-only安装

1. 解压缩下载文件夹，将`./boost_1_71_0/boost`文件夹放置于`SOMEWHERE`目录下，个人使用的`SOMEWHERE`如下
   * win: `C:\Users\zchao\Documents\software\include`
   * bash: `/home/zhangc/cpp_third_party_include`
2. `SOMEWHERE`目录下的文件结构大致如下
   * `SOMEWHERE/boost/accumulators`
   * `SOMEWHERE/boost/algorithm`
   * `SOMEWHERE/boost/align`
   * ...
3. 修改environment variable，**务必**将`SOMEWHERE`换成自己的目录
   * powershell: `$env:CPP_THIRD_PARTY_INCLUDE="SOMEWHERE"`
   * bash: `export CPP_THIRD_PARTY_INCLUDE="SOMEWHERE"`

cmake安装

1. 设置环境变量 `Boost_ROOT`
   * powershell: `$env:Boost_ROOT+="C:/Users/zchao/Documents/software/boost/boost_1_71_0"`
   * bash: `export Boost_ROOT="/home/zhangc/software/boost/boost_1_71_0"`
2. 见`ws_cmake`
3. 在`ws_cmake/build`路径下`cmake -S .. -B .`
4. 在`ws_cmake/build`路径下`cmake --build .`

## minimum working example

使用「header-only安装方式」，假设已经设置好了`CPP_THIRD_PARTY_INCLUDE`环境变量

`draft00.cpp`

```cpp
#include <iostream>
#include <boost/format.hpp>
int main(int argc, char const *argv[])
{
    std::cout << boost::format("hello %1%") % "world\n";
    return 0;
}
```

1. 编译
   * powershell: `g++ draft01.cpp -std=c++11 -o tbd00.exe -I $env:CPP_THIRD_PARTY_INCLUDE`
   * bash: `g++ draft01.cpp -std=c++11 -o tbd00.exe -I $CPP_THIRD_PARTY_INCLUDE`
2. 运行`./tbd00.exe`
