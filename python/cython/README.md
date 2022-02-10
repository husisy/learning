# cython

1. link
   * [official site](https://cython.org/)
   * [documentation](https://cython.readthedocs.io/en/latest/)
   * [github wiki](https://github.com/cython/cython/wiki)
   * [stackoverflow/extending-setuptools-extension-to-use-cmake](https://stackoverflow.com/a/48015772)
2. install `conda install -c conda-forge cython`
   * [MS/Visual-Studio](https://visualstudio.microsoft.com/visual-cpp-build-tools/), [stackoverflow](https://stackoverflow.com/a/67033876), [python compiler for windows](https://blogs.msdn.microsoft.com/pythonengineering/2016/04/11/unable-to-find-vcvarsall-bat/)
   * win需要下载 Visual Studio Community, Desktop development with cpp
3. keyword `cdef, cpdef`
4. 文档提及：将所有变量加上类型声明甚至可能会变慢（不必要的类型转换，buffer unpacking）
5. 生成html网页查看转换情况
   * `Cython.Build.cythonize('my_cython_pkg.pyx', annotate=True)`
   * `pip install -e .`, `pip install --use-feature=in-tree-build .`
6. `cdef int p[1000]`申请的是call stack空间，不能用于存放长数组 [cython/memory-allocation](https://cython.readthedocs.io/en/latest/src/tutorial/memory_allocation.html#memory-allocation) [cython/python-array](https://cython.readthedocs.io/en/latest/src/tutorial/array.html#array-array) [cython/numpy-array](https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html#memoryviews)

`zip_safe=False`

`# distutils: language=c++`

TODO

1. [cython-fused-types](https://cython.readthedocs.io/en/latest/src/userguide/fusedtypes.html)
2. [cython-interfacing-with-external-c-code](https://cython.readthedocs.io/en/latest/src/userguide/external_C_code.html)
3. [github/cython-sse-example](https://github.com/Technologicat/cython-sse-example)

## mwe00 (ws00)

文件结构

```bash
ws00
├── python
│   └── my_cython_pkg.pyx
├── setup.py
└── draft00.py
```

1. 安装 `pip install .`
   * `python setup.py build_ext --inplace`
2. 卸载 `pip uninstall my_cython_pkg`
3. 运行 `python draft00.py`

## mwe02 (ws02)

[cython-using-c-libraries](https://cython.readthedocs.io/en/latest/src/tutorial/clibraries.html)

1. 安装 `pip install .`
2. 卸载 `pip uninstall my_cython_pkg`
3. 运行 `python draft00.py`

## mwe03 (ws03)

[cython-for-numpy-users](https://cython.readthedocs.io/en/latest/src/userguide/numpy_tutorial.html)

## jupyter notebook

```Python
%load_ext Cython
```

```cython
%%cython

cdef int a = 0
for i in range(10):
    a += i
print(a)
```
