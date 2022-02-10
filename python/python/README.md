# Python

当前文件夹介绍python编程语言自身的特性及标准库（部分较复杂的标准库单独成立文件夹）

1. link
   * [廖雪峰-Python教程](https://www.liaoxuefeng.com/wiki/1016959663602400)
   * [python documentation / tutorial](https://docs.python.org/3/tutorial/index.html)
   * [python compiler for windows](https://blogs.msdn.microsoft.com/pythonengineering/2016/04/11/unable-to-find-vcvarsall-bat/)
   * [time complexity](https://wiki.python.org/moin/TimeComplexity?)
   * [python documentation / built-in functions](https://docs.python.org/3/library/functions.html#built-in-functions)

## quickstart-搭建python运行环境

偏见：使用`miniconda`，不使用linux系统`python`

1. 下载
   * windows：[下载链接](https://docs.conda.io/en/latest/miniconda.html)
   * linux: `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`
2. 安装
   * windows: 双击run run run
   * linux: `bash Miniconda3-latest-Linux-x86_64.sh`, run run run，安装完之后必须重新登录黑框框miniconda才会生效
3. （可选）境内使用conda镜像
   * [清华大学镜像](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/)
4. 启动miniconda
   * windows：开始菜单的`anaconda prompt`应当作为conda环境的唯一入口
   * linux：登录黑框框即可
5. （可选）创建环境`python_env`，**不建议**使用与修改`base`环境
   * 创建环境`conda create -n python_env`
   * 安装包`conda install -n python_env -c conda-forge python ipython scipy`
   * 激活环境`conda activate python_env`
   * 运行: `python`或者`ipython`，推荐`ipython`

更多的环境搭建见`python/conda_xxx/README.md`目录

## Python Guide - project repository structure

1. link
   * [github](https://github.com/realpython/python-guide/)
   * [site](https://docs.python-guide.org/)
   * [PEP20 The Zen of Python](https://www.python.org/dev/peps/pep-0020/)
2. text editor and IDE: `Vim`, `Emacs`
3. project name: `MyProject`
4. `./MyProject/` or `./MyProject.py`
   * avoid circular dependencies
   * avoid hidden coupling
   * avoid heavy usage global state or context
   * avoid spaghetti code and ravioli code
5. `./LICENSE`: see [choose license](https://choosealicense.com/)
6. `./setup.py`: package and distribution management
7. `./requirements.txt` (for pip installation) or `./environment.yml` (for conda dependency)
8. `./doc/`
9. `/test/`
10. `./Makefile`
11. regarding django application, see [python-guide-django](https://docs.python-guide.org/writing/structure/#regarding-django-applications)
12. pure function
    * deterministic: given a fixed input, the output will be the same
    * easier to refactored or optimized
    * easier to build unittest (less need for complex context setup and data cleaning afterwards)
    * easier to manipulate, decorate and pass around
13. throwaway variables use double underscore `__` or single underscore `_`

## PEP8 Style Guide for Python Code

1. link
   * [PEP8 Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)
   * [PEP8.org](https://pep8.org/)
   * [PEP7 Style Guide for C Code](https://www.python.org/dev/peps/pep-0007/)
   * [PEP257 Docstring Convention](https://www.python.org/dev/peps/pep-0257/)
2. 一致性consistency
3. 使用4 space而非tab
4. 尽量避免使用转义字符来断行（存在特例`with a() as a, b() as b`）
5. triple-quoted strings使用double quote character而非single quote character

```Python
"""This is the example module.

This module does stuff.
"""

from __future__ import barry_as_FLUFL

__all__ = ['a', 'b', 'c']
__version__ = '0.1'
__author__ = 'Cardinal Biggles'

import os
import sys
```
