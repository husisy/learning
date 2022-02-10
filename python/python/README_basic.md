# python basic

1. link
   * [official site](https://www.python.org/)
   * [The Python Standard Library](https://docs.python.org/3/library/index.html#library-index)
   * [The Python Language Reference](https://docs.python.org/3/reference/index.html#reference-index)
   * [Glossary](https://docs.python.org/3/glossary.html#glossary)
   * [廖雪峰的官方网站](https://www.liaoxuefeng.com/wiki/0014316089557264a6b348958f449949df42a6d3a2e542c000)
   * [Python 入门指南](http://www.pythondoc.com/pythontutorial3/)
   * [the Python tutorial](https://docs.python.org/3/tutorial/)
2. install: 见`python/conda/README.md`
3. import package: 见`python/python_module.md`
4. Python interpreter: `CPython IPython PyPy Jython IronPython`
5. 交互模式
   * 命令行模式 `sys.md`
   * Python交互模式
   * `Jupyter notebook.md`
6. 应用场景：文本文件处理，文件整理，小型的自定义数据库，GUI应用程序，小游戏
7. `_`变量对于用户只读。"给它赋值"只会创建独立的同名局部变量，屏蔽了系统内置变量的魔术效果
8. [built-in functions](https://docs.python.org/3/library/functions.html)，常用函数`print() len() type() max() abs() range() isinstance()`
9. 数据类型
   * `int`: `0x`
   * `float`:`1.2e-5`
   * `str`: `'abc'`
   * `bool`: `True`,`False`, `and`, `or`, `not`
   * `None`
   * `bytes`: `b'ABC'`
   * `3+5j`, `5+3i`
10. 运算符`/`, `//`(地板除), `%`, etc
11. str
    * `"abc"`, `'''...'''`(Ellipsis转换为`'\n'`), `'\t'`, `'\\'`, `'\''`, `r'\'`
    * ASCII, GB2312, Shift_JIS, Euc-kr, Unicode, UTF-8
    * 计算机内存中：Unicode编码
    * 保存到硬盘或者需要传输：UTF-8编码。
    * `ord('中') #20013`
    * `chr(25991) #文`
    * `'ABC'.encode('ascii') #b'ABC'`
    * `'中文'.encode('utf-8') #b'\xe4\xb8\xad\xe6\x96\x87'`
    * `b'ABC'.decode('ascii') #'ABC'`
12. keyword
    * `and del from not while as elif global or  with`
    * `assert else if pass yield break except import print`
    * `class exec in raise continue finally is return def for lambda try`
13. 流程控制
    * `if...elif...else`
    * `for...in...`
    * `while`
14. `tuple list set dict slice zip iter next`
15. 迭代`collections.Iterable collections.Iterator`
16. 生成器 generator `(x * x for x in range(10))`, `yield`

## list

1. `z1 = [bula,bula]`
2. indexing: `z1[ind]`
   * start from `0`
   * only integer or `slice`, basic indexing of `numpy`
   * negative index
3. method
   * `z1.append()`
   * `z1.insert(1)`
   * `z1.pop(0)`
4. empty list `[]`
5. `for ind1, tmp1 in enumerate(['a','b','c'])`
6. 列表生成式 List Comprehension
   * `[x*x for x in range(1,11) if x%2==0]`
   * `[m + n for m in 'ABC' for n in 'XYZ']`

## tuple

1. `(1)` and `(1,)`
2. empty tuple `()`
3. 不可变tuple：`(1,2,[3])`
4. assign value: `a,b,c=1,2,3`
5. `list((1,2,3))`
6. `list(range(3))`

## dict

1. `z1 = {'Michael': 95, 'Bob': 75, 'Tracy': 85}`
2. `'a' in a`
3. key不可变
4. dict也是可迭代对象，默认迭代key
   * `for ind1 in z1.values()`
   * `for ind1,ind2 in z1.items()`
5. method
   * `get()`
   * `pop()`
   * `setdefault()`
6. `dict(zip('abc',range(3)))`

```python
z1 = {'jack':4098, 4098:'jack'}
z1['jack4098'] = '4098jack'
del z1['jack4098']
list(z1.keys())
'jack4098' in z1
z1 = dict([('jack',4098),(4098,'jack')])
z1 = dict(jack=4098)
{x:x**2 for x in (2,3,4)}
for k,v in z1.item():
for q, a in zip(['name','quest','faveorite color'], ['lancelot','the holy grail','blue']):
```

## set

1. `z1 = set([1, 1, 2, 2, 3, 3])`
2. method
   * `add()`
   * `remove()`
3. operation
   * `|`
   * `&`
4. key不可变

## function

1. 函数名赋值：`z1 = abs`
2. `def hf1(): print('hello world')`
3. 缺省返回值`None`
4. 位置参数
5. 默认参数
   * `def hf1(x=1,y=2): print(x+y)`
   * `hf1(y=0)`
   * 可不按顺序提供部分参数，**位置参数亦可使用这一特性？**
   * 默认参数在位置参数之后
   * 默认参数必须指向不变对象
6. 可变参数
   * `def hf1(*args): return sum(args)`
   * `hf1(*(1,2,3,4))`
   * 输入参数组装为tuple
7. 关键字参数
   * `def hf1(**kw): print(kw)`
   * `hf1(a=1,b=2)`
   * `hf1(**{'a':1,'b':2})`
   * 输入参数组装为dict
8. 命名关键字参数
   * `def hf1(*,z1=1,z2=2): print(z1+z2)` 特殊分隔符
   * `def hf1(*args,z1,z2): print('hello world')` 可变参数后不需要特殊分隔符
   * 必须传入参数名
   * 可给出缺省值，从而不提供参数
9. 参数顺序：必选参数、默认参数、可变参数、命名关键字参数、关键字参数
10. `func(*args, **kw)`
11. 递归函数
    * 尾递归优化

## Higher-order function

1. 让函数的参数能够接收别的函数
2. sort
   * `sort()`: in-place, `list` only
   * `sorted()`: out-place, for `iterable`
   * `sorted("This is a test string from Andrew".split(), key=str.lower)`
3. `list(map(str,[1,2,3,4,5,6,7,8,9,10]))`
4. `from functo0ls import reduce`
   * `def char2num(s): return {'0':0,'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9}[s]`
   * `reduce(lambda x,y: x*10+y, map(char2num,'13593'))`
5. `filter()`
6. 返回函数和闭包
7. 匿名函数`lambda`
   * `hf1 = lambda x=4:x>3`
   * `[x for x in range(10) if hf1(x) and x%2]`
8. 函数属性
   * `hf1.__name__`
9. 装饰器decorator
   * `import functools`
10. 偏函数 partial function
    * `import functools`
    * `int2 = functools.partial(int,base=2)`

```python
import functools

def log(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        print('call %s():' % func.__name__)
        return func(*args, **kw)
    return wrapper
@log
def now():
    print('2015-3-25')
# now = log('execute')(now)

def log(text):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kw):
            print('%s %s():' % (text, func.__name__))
            return func(*args, **kw)
        return wrapper
    return decorator
@log('execute')
def now():
    print('2015-3-25')
# now = log('execute')(now)
```

## example-basic

```python
1+2
exit()
print('hello world')
z1 = input()
# comment1
''' comment2 '''
""" comment3 """

# operator
# + - *
17/3
17//3
17%3
3**2
2**3
print(_)
```
