from urllib.request import urlopen
import contextlib


class MyClass1(object):
    def __init__(self):
        print('MyClass1.__init__()')
    def __enter__(self):
        print('MyClass1.__enter__()')
        return self
    def __exit__(self, exc_type, exc_value, traceback):
        print('MyClass1.__exit__({}, {}, {})'.format(exc_type, exc_value, traceback))
        if exc_type:
            print('Error')
    def hf1(self):
        print('MyClass1.hf1()')
with MyClass1() as x1:
    x1.hf1()


class MyClass2(object):
    def __init__(self):
        print('MyClass1.__init__()')
    def hf1(self):
        print('MyClass1.hf1()')
@contextlib.contextmanager
def hf1():
    print('__main__.hf1() start')
    x1 = MyClass2()
    yield x1
    print('__main__.hf1() end')
with hf1() as x1:
    x1.hf1()


@contextlib.contextmanager
def hf2():
    print('__main__.hf2() start')
    yield
    print('__main__hf2() end')
with hf2():
    print('hello world')


with contextlib.closing(urlopen('https://www.python.org')) as response:
    print(type(response), response.status)
