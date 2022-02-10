# json

1. link
   * [github](https://github.com/nlohmann/json)

## installation

header-only安装

1. 解压缩下载文件夹，将`./include/nlohmann`文件夹放置于`SOMEWHERE`目录下，个人使用的`SOMEWHERE`如下
   * win: `C:\Users\zchao\Documents\software\include`
   * bash: `/home/zhangc/cpp_third_party_include`
2. `SOMEWHERE`目录下的文件结构大致如下
   * `SOMEWHERE/nlohmann/json.hpp`
   * `SOMEWHERE/nlohmann/json_fwd.hpp`
   * `SOMEWHERE/nlohmann/thirdparty/`
   * ...
3. 修改environment variable，**务必**将`SOMEWHERE`换成自己的目录
   * powershell: `$env:CPP_THIRD_PARTY_INCLUDE="SOMEWHERE"`
   * bash: `export CPP_THIRD_PARTY_INCLUDE="SOMEWHERE"`

cmake安装

1. `BOOST_ROOT`
2. *TODO*

## minimum working example

使用「header-only安装方式」，假设已经设置好了`CPP_THIRD_PARTY_INCLUDE`环境变量

`draft00.cpp`

```cpp
#include <iostream>
#include <nlohmann/json.hpp>
int main(int argc, char const *argv[])
{
    nlohmann::json z0;
    z0["pi"] = 3.1415926535;
    z0["happy"] = true;
    z0["name"] = "Niels";
    std::cout << z0;
    return 0;
}
```

1. 编译
   * powershell: `g++ draft01.cpp -std=c++11 -o tbd00.exe -I $env:CPP_THIRD_PARTY_INCLUDE`
   * bash: `g++ draft01.cpp -std=c++11 -o tbd00.exe -I $CPP_THIRD_PARTY_INCLUDE`
2. 运行`./tbd00.exe`
