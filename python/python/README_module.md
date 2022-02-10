# python module

1. link
   * [廖雪峰的官方网站](https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/0014318447437605e90206e261744c08630a836851f5183000)
   * [python学习笔记之module && package](http://arganzheng.iteye.com/blog/986301)
   * [知乎-Python中的模块、库、包有什么区别](https://www.zhihu.com/question/30082392)
   * [python项目内import其他内部package的模块的正确方法](http://blog.csdn.net/luo123n/article/details/49849649)
   * [python documentation](https://docs.python.org/3/tutorial/modules.html#packages)
2. 自我包含并且有组织的代码片段；一个 .py 文件就是个 module。
3. module代码的第一个字符串被视为模块的文档注释
4. special variable: `__all__ __name__ __author__ __doc__`
5. private variable: `_xxx __xxx`
6. import module: `import moduleA; moduleA.objectB`, `from modduleA import objectB`
7. install module: see python/conda
8. package查找路径`sys.path`
9. 添加路径: `sys.path.append('bula')`, 环境变量`PYTHONPATH`
10. package: 一个有层次的文件目录结构（特殊的module），定义了由n个模块或n个子包组成的python应用程序执行环境。即一个包含__init__.py 文件的目录，该目录下有`__init__.py`, module以及subpackage。subpackage是指含有```__init__.py```的子目录
11. intra package references: `from . import xxx`, `from .. import xxx`, `from ..xx import xxx`

```bash
Package1
    __init__.py
    Module1.py
    Module2.py
    Subpackage1
        __init__.py
        Module1.py
        Module2.py
```

```python
from Package1 import Module1
from Package1 import Subpackage1
import Packag1.Module1
import Packag1.Subpackage2
```
