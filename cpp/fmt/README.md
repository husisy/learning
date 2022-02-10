# fmt

1. link
   * [github](https://github.com/fmtlib/fmt)
   * [documentation](https://fmt.dev/latest/index.html)

## minimum working example ws00

文件结构

```bash
.
include/
|--fmt/
   |--core.h
   |--format.h
   |--format-inl.h
src/
|--format.cc
draft00.cpp
```

其中

1. `include/fmt/*`来自于fmt下载源文件`include/*`
2. `src/format.cc`来自于fmt下载源文件`src/format.cc`

`draft00.cpp`

```cpp
#include <iostream>
#include "fmt/format.h"
int main(int argc, char const *argv[])
{
    std::cout << fmt::format("hello {}.", "world") << std::endl;
    return 0;
}
```

编译运行

1. `g++ draft00.cpp src/format.cc -std=c++11 -I ./include -o tbd00.exe`
2. `./tbd00.exe`

## cmake minimum working example - ws00

**WARNING**: cmake-generator `Visual Studio 16 2019`失败

cmake-generator `MinGW Makefiles`通过

```bash
mkdir build
cd build
cmake -S .. -B . -G "MinGW Makefiles" #default generator failed
cmake --build .
cmake --install . --prefix C:/Users/zchao/Documents/software/fmt
$env:fmt_ROOT="C:\Users\zchao\Documents\software\fmt"
```
