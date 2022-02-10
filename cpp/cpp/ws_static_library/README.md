# 静态链接

1. 第三方生成静态链接库
   * `g++ -c third_party/utils.cpp -o third_party/utils.o` 先生成目标文件
   * `mkdir static_library`
   * `ar -crv static_library/libutils.a third_party/utils.o`
2. 使用者编译时指定静态链接库
   * `g++ -o tbd00.exe draft00.cpp -L ./static_library -l utils`
   * 此时使用者能够获得静态链接库与头文件，但不能获得第三方源码或者目标文件
