# 动态链接-linux

1. 第三方生成动态链接库
   * `mkdir dynamic_library`
   * `g++ -fPIC -shared -o dynamic_library/libutils.so third_party/utils.cpp`
   * 添加至环境变量`LD_LIBRARY_PATH`, `export LD_LIBRARY_PATH="$PWD/dynamic_library:$LD_LIBRARY_PATH"`
2. 使用者编译时指定动态链接库`g++ -o tbd00.exe draft00.cpp -L dynamic_library -l utils`
