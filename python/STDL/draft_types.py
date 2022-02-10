import types

types.FunctionType #typing.Callable
types.BuiltinFunctionType
types.LambdaType
types.GeneratorType #collections.abc.Generator
types.MethodType


# create class using type()
class MyClass0(object):
    def hello(self):
        print('hello from class MyClass0')
print(MyClass0, type(MyClass0))
MyClass0().hello()

def hf1(self):
    print('hello from function hf1')
MyClass1 = type('MyClass1', (object,), {'hello':hf1})
print(MyClass1, type(MyClass1))
MyClass1().hello()


# override operator
class MyClass2(object):
    def __init__(self, a):
        self.a = a
    def __add__(self, x):
        return self.a + x.a
print(MyClass2(2) + MyClass2(3))
