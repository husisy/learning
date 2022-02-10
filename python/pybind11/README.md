# pybind11

1. install `conda install -c conda-forge pybind11`
2. 可运行示例
   * [github-pybind-cmake-example](https://github.com/pybind/cmake_example)

## cmake installation

1. 预先安装
   * cmake，见独立文档
   * boost，见独立文档
2. 下载pybind11-github至目录`SOMEWHERE`，**务必**将`SOMEWHERE`替换为自己的目录
3. 可能需要在`SOMEWHERE/test/CMakeLists.txt`中添加`cmake_policy(SET CMP0074 NEW)`
4. 创建目录`SOMEWHERE/build`
5. 在`SOMEWHERE/build`目录下执行`cmake -S .. -B .`
6. 在`SOMEWHERE/build`目录下下执行`cmake --build .`
7. 在`SOMEWHERE/build`目录下下执行：**务必**将`ANYWHERE`替换为自己的目录
   * windows: `cmake --install . --prefix ANYWHERE/pybind11`
   * linux: `cmake --install . --prefix ANYWHERE/pybind11`
8. 设置环境变量：**务必**将`ANYWHERE`替换为自己的目录
   * windows: `$env:Pybind11_ROOT="ANYWHERE/pybind11"`
   * linux: `export Pybind11_ROOT="ANYWHERE/pybind11"`
9. 至此，便可在CMakeLists.txt文件中使用`find_package(pybind11 REQUIRED)`

## minimum working example - mwe00

原始代码仓库：[github-pybind-python-example](https://github.com/pybind/python_example)

1. 所有文件
   * `mwe00/setup.py`
   * `mwe00/src/main.cpp`
2. 安装该Python包
   * 在`mwe00/`目录下执行`pip install .`
   * 如果之前已安装该包，该条命令会先卸载已有的包再安装
3. 运行
   * 在命令行执行`python -c "from zctest_module00 import add; print(add(2,33))"`
   * 在python交互式环境下执行`from zctest_module00 import add; print(add(2,33))`
   * module name需要在两处指定，在`setup.py/Extension('zctest_module00', ...)`，在`src/main.cpp/PYBIND11_MODULE(zctest_module00, m)`，`src/main.cpp/m.doc()`处不是必须
4. 卸载该Python包
   * `pip uninstall zctest_pybind11`或者`pip uninstall zctest-pybind11`
   * 在`mwe00/setup.py/setup(name='zctest_pybind11',...)`处指定了该package的名字

TODO

1. 该仓库中包含sphinx文档生成示例，同时见`main.cpp`
2. 该仓库包含conda使用示例
3. 该仓库支持darwin操作系统

## minimum working example - submodule

`ws_submodule`

## workspace cmake

1. submodule问题
   * `setup.py`中应该为`CMakeExtension('zctest_pybind11._cpp')`
   * `CMakeLists.txt`中应该为`project(whatever)`, `pybind11_add_module(_cpp main.cpp)`
   * `main.cpp`中应该为`PYBIND11_MODULE(_cpp, m)`

## TODO

1. pybind11-github/test目录下有大量参考栗子
2. 如何与单元测试集成
3. cmake栗子，看文档
4. 复杂编译链如何操作，看文档
5. 如何进一步封装，见[github-matplotlib-mplcairo](https://github.com/matplotlib/mplcairo)
6. 如何减少数据传来传去，看doc-numpy-example
7. 栗子[github-qulacs](https://github.com/qulacs/qulacs)
