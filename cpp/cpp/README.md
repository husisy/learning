# cplusplus

文档说明

1. 本文件中所有backslash写作slash
2. 当前文件夹所有文件请用`gcc/g++`编译
   * winOS的`cl.exe`与下文的编译参数差别太大，请用[mingw-w64-g++](http://mingw-w64.org/doku.php)入门
   * 之后过渡到cmake来实现兼容性

个人对于`c++`偏见声明

1. 标准委员会指定的标准乱七八糟
   * 参见[知乎-cpp17问题](https://www.zhihu.com/question/56943731)
   * 参见[cpp-random-stl](https://en.cppreference.com/w/cpp/header/random)：一个基础运算功能封装成了`device/seed/engine/distribution`四层抽象，而且四层抽象依旧没法很好地实现「调用时随机/编译后随机」（需进一步hack标准库中的`ctime`库以及添加`static`特性）；过分冗余的抽象，且未解决用户使用时的繁琐
   * `PI/Module/range`等特性一拖再拖
2. 整个社区停滞不前
   * 第三方库只能通过「single-header」文件的方式来降低用户的学习成本
   * 第三方库文档严重不足
   * `cxxopt/unittest`等第三方库在提供接口的同时，暴露出大量的实现细节（复杂的类型系统），缺失通用第三方库
   * `cmake`等构建工具学习难，缺失pip对标工具

基于前言中的所述的「偏见」

1. `c++`标准选择
   * 一律使用`c++11`，不向前兼容
   * 禁止使用`c++17`，编译器支持不理想，增加的特性无关紧要
   * 不期待`c++20` @2020
2. **禁止**使用`size_t`以及`std::size_t`：通常`size_t`定义为`unsigned int`，如下论述中将用`unsigned int`代指`size_t`
   * 通过类型系统来保证值非负是没有意义的，如果有意义，那同样也需要“非零类型”、“非正类型”、“恒正类型”、“恒负类型”、“非1类型”等
   * 负值长度是坏定义的，但「禁止`size_t`」不等价于「取消负值检查」，在我看来「负值检查」应该是调用者或者被调函数的责任
   * 负值索引是有意义的，参见numpy索引规则
   * `unsigned int`长度是没有意义的：诚然使用`unsinged int`的确可以获得更大长度的数组，但一个如此长以致于无法用`int`去索引，那这个长度又有什么意义呢
   * 根据numpy的使用经验，「下标索引」与「值」不可避免地要混用，那么「区分前者为`unsigned int`后者为`int`」就是没有意义的
   * 我不会给“摩托罗拉68000”写代码，你多半也不会

概述

1. link
   * [高速上手C++11/14/17](https://github.com/changkun/modern-cpp-tutorial)1. 默认使用标准`c++11`，不打算向前兼容，暂不使用`c++14/c++17/c++20`特性
   * [github/CppCoreGuidelines-zh-CN核心指南](https://github.com/lynnboy/CppCoreGuidelines-zh-CN)
2. 显示返回结果
   * win-cmd: `echo %ERRORLEVEL%`
   * powershell: `echo $LASTEXITCODE`
   * linux-bash: `echo $?`
   * `-1`通常作为错误代码
3. 文件结束符End Of File
   * windows: `ctrl+z`
   * linux: `ctrl+d`
4. 常见错误：语法错误Syntax，类型错误type，声明错误declaration
5. 头文件
   * `#include <iostream>`: 标准库
   * `#include "test.h"`: 非标准库头文件
6. 术语：实参argument，形参parameter，赋值assignment，程序块block，缓冲区buffer，内置类型built-in type，字符串常量string literal，花括号curly brace，编辑edit-编译compile-调试debug，表达式expression
7. 选择算术类型
   * 不可能为负时，选择无符号类型
   * 正数运算用int，超过int范围时用long long
   * 浮点运算用double

## minimum working example-00

`draft00.cpp`

```cpp
#include <iostream>
int main(int argc, char *argv[])
{
    std::cout << "hello world" << std::endl;
    return 0;
}
```

1. 编译`g++ draft00.cpp -std=c++11 -o tbd00.exe`
2. 运行 `./tbd00.exe`

cpp的浮点数向整数转型总是向`0`舍入

```cpp
#include <iostream>
int main(int argc, char *argv[])
{
    double x=1.2,y=0.8,z=0.5,a=-0.5,b=-0.8,c=-1.2;
    std::cout << (int)x << "," << (int)y << "," << (int)z << std::endl; //1,0,0
    std::cout << (int)a << "," << (int)b << "," << (int)c << std::endl; //0,0,-1
    return 0;
}
```

奇奇怪怪的整数取模运算

```cpp
#include <iostream>
int main(int argc, char *argv[])
{
    std::cout << ((-1)%3) << "," << (2%3) << std::endl; //-1,2
    return 0;
}
```

## ws_link

静态链接见`ws_link`

动态链接linux见`ws_shared_library_linux`

动态链接-windows *待完善*
