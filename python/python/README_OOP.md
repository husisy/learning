# python OOP

1. lin
   * [deep thoughts by Raymond Hettinger](https://rhettinger.wordpress.com/2011/05/26/super-considered-super/), best practice to use `super()`
   * [廖雪峰的官方网站](https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000/0014318645694388f1f10473d7f416e9291616be8367ab5000)
   * [stackoverflow / what does super do in Python](https://stackoverflow.com/q/222877/7290857)
2. `class Student(object): pass`
3. 绑定属性，`z1 = Student()`, `
   * 实例绑定属性：`z1.bula = 1`
   * class绑定属性：`Student.bula = 1`
   * class绑定方法：`Studeng.bula = lambda x:x**2`
4. `def __init__(self,bula):`
5. 访问限制
   * 私有变量：`__bula`，不能直接访问，改名为`_Student__bula`
   * 某些变量：`_bula`，不要随意访问
   * 特殊变量：`__bula__`, `z1.__len__()` = `len(z1)`
6. 继承与多态
   * 继承：subclass调用class的方法
   * 多态：subclas覆盖class的方法
7. 获取对象信息
   * `import types`
   * `types.FunctionType`, `types.LambdaType`, `types.GeneratorType`
   * `isinstance()`
   * `dir()`
   * `hasattr()`, `getattr()`, `setattr()`
8. 开闭原则：对拓展开放，对修改封闭
9. 静态语言，动态语言（鸭子类型）
10. 类属性
    * `name='Student'`，通过`Student.name`或者`z1.name`访问（如果实例没有覆盖该属性）
    * `self.name='Student'`
11. 删除实例属性：`del z1.name`
12. 限制实例属性：`__slots__=('name', 'age')`，继承无效
13. `@property`将方法变成属性 **TBA**
    * 限定只读
14. 多重继承 `MixIn`
15. 定制类
    * `__slots__`
    * `__len__`
    * `__name__`
    * `__str__`
    * `__repr__`: 一般`__repr__=__str__`
    * `__iter__`
    * `__getitem__`
    * `__setitem__`
    * `__delitem__`
    * `__getattr__`
    * `__call__`, `callable`
16. 枚举类 `from enum import Enum`
17. 元类 metaclass **一脸懵逼**

```Python
class Student(object):
    @property
    def score(self):
        return self._score

    @score.setter
    def score(self, value):
        if not isinstance(value, int):
            raise ValueError('score must be an integer!')
        if value < 0 or value > 100:
            raise ValueError('score must between 0 ~ 100!')
        self._score = value
```
