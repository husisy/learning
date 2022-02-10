# README project

1. link
   * [github](https://github.com/realpython/python-guide/)
   * [site](https://docs.python-guide.org/)
   * [PEP20 The Zen of Python](https://www.python.org/dev/peps/pep-0020/)
2. text editor and IDE
   * Vim
   * Emacs

该文档展示一个常见的项目开发周期

1. start
2. add module file
3. add package
4. add more functions
5. add setuptools
6. add examples

以下用`wnyl`来作为项目名，为whatever name you like缩写

## start

文件结构

```bash
.
├── draft00.py
├── README.md
├── .gitignore
├── .git/
├── _developer/
└── data/
```

文件说明

1. `draft00.py`是草稿代码，想到什么就写下什么，不急于实现的功能用`#TODO`标志
2. `README.md`：markdown文档，用于记录一些有用的信息，例如提醒自己的备注、帮助他人快速了解项目的说明文字、开发环境配置说明等
3. 其它（下文不再赘述）
   * `.gitignore`和`.git/`：建议在项目初期就开始使用git进行管理
   * `_developer`：不适宜git管理的文件夹，例如PPT/pdf等文件
   * `data/`：数据文件，例如需要备份的计算结果、绘图文件（例如用于`README.md`）
   * 可选：`project_dependency.ini`, `project_dependency.ini.example`, `project_dependency.config`

## add module file

文件结构

```bash
.
├── draft00.py
├── utils.py
├── test_utils.py
└── README.md
```

文件说明

1. `from utils import xxx`
2. `utils.py`：将`draft00.py`文件中将频繁复用以及功能基本固定的函数放至`utils.py`文件
3. `test_utils.py`：使用`pytest`模块对`utils.py`文件中的函数进行单元测试

## add package

文件结构

```bash
.
├── draft00.py
├── utils.py
├── test_wnyl.py
├── README.md
└── wnyl/
    ├── __init__.py
    └── _utils.py
```

文件说明

1. `from wnyl import xxx`
2. `./wnyl/__init__.py`：python package必须文件，该文件中应包含`wnyl`模块所有的公开方法，例如需要将`./wnyl/_utils.py`中的所有方法导入至`__init__.py`中
3. `./wnyl/_utils.py`：将`./utils.py`中与`wnyl`密切相关的函数拧出来放在该文件中
   * 该package内部使用的模块，**禁止**在外部导入该模块
4. `./test_wnyl.py`：对`wnyl`模块进行单元测试，不再对`./utils.py`进行测试
5. `./utils.py`：仍保留`draft00.py`需要使用的函数，但这些函数与`wnyl` package应该关系不大

## add more functions

文件结构

```bash
.
├── draft00.py
├── utils.py
├── README.md
├── tests/
│   ├── test_utils.py
│   └── test_yyy.py
└── wnyl/
    ├── __init__.py
    ├── yyy.py
    └── _utils.py
```

文件说明

1. `wnyl/yyy.py`：向wnyl中添加了更多的功能
2. `tests/test_utils.py`：将`test_utils.py`放在单独的文件夹`tests`
3. `tests/test_yyy.py`：对`wnyl/yyy.py`进行单元测试

## add setuptools

文件结构

```bash
.
├── draft00.py
├── utils.py
├── setup.py
├── README.md
├── tests/
│   ├── test_utils.py
│   └── test_utils.py
└── python/
    └── wnyl/
        ├── __init__.py
        ├── xxx.py
        └── _utils.py
```

文件说明

1. `pip install -e .`或者`pip install .`, `from wnyl import xxx`
2. `setup.py`：使用`pip install`必须文件
3. `./python`：将`wnyl`文件夹放至`./python`目录下，否则在`draft00.py`中会优先导入当前目录的`wnyl`而非`pip install`的`wnyl`

## add examples

文件结构

```bash
.
├── setup.py
├── README.md
├── examples/
│   └── draft00.py
├── tests/
│   ├── test_utils.py
│   └── test_utils.py
└── python
    └── wnyl/
        ├── __init__.py
        ├── xxx.py
        └── _utils.py
```

文件说明

1. `examples/draft00.py`：将原本的`./draft00.py`以及`./utils.py`整理为用户教程放置于`examples/`目录
2. `./draft00.py`与`./utils.py`建议删除，但也可以留作开发者使用
3. 同步至代码托管仓库（例如github/gitee），安装方式可以修改为`pip install git+https://github.com/husisy/RGF.git`

## Project Repository Structure

1. project name: `MyProject`
2. `./MyProject/` or `./MyProject.py`
   * avoid circular dependencies
   * avoid hidden coupling
   * avoid heavy usage global state or context
   * avoid spaghetti code and ravioli code
3. `./LICENSE`: see [choose license](https://choosealicense.com/)
4. `./setup.py`: package and distribution management
5. `./requirements.txt` (for pip installation) or `./environment.yml` (for conda dependency)
6. `./doc/`
7. `/test/`, see [testing your code](https://docs.python-guide.org/writing/tests/)
8. regarding django application, see [python-guide-django](https://docs.python-guide.org/writing/structure/#regarding-django-applications)
9. pure function
   * deterministic: given a fixed input, the output will be the same
   * easier to refactored or optimized
   * easier to build unittest (less need for complex context setup and data cleaning afterwards)
   * easier to manipulate, decorate and pass around
10. throwaway variables use double underscore `__` or single underscore `_`
11. [time complexity](https://wiki.python.org/moin/TimeComplexity?)
