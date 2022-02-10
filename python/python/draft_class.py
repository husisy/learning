import types
import pickle

class MyClass00(object):

    def __init__(self, prop0):
        self.prop0 = prop0
        self.__private_property = 233
        self.__not_private_property__ = 2333 #NOT recommend
        self._serious_property = 2333 #use it IF AND ONLY IF you know it clearly

    def method00(self):
        ret = self.__private_property + 233
        return ret

    def set_prop0(self, value):
        # equivalent to "xxx.prop0 = value"
        assert isinstance(value, int)
        self.prop0 = value

    def __getattr__(self, attr):
        if attr == 'prop0':
            return self.prop0
        if attr == 'prop0_for_fun':
            return self.prop0
        raise AttributeError('"MyClass00" has no attribute {}'.format(attr))


z0 = MyClass00(23)
# hasattr(z0, '_Class00__private_property') #NEVER use this
# z0.__not_private_property__ #NOT recommend
z0.prop0
z0.method00()
z0.set_prop0(233)
z0.prop0 = 23333
z0.bad_prop = '233' #NOT recommend
z0.prop0_for_fun
getattr(z0, 'prop0_for_fun')
hasattr(z0, 'prop0_for_fun') #True
# setattr

hf0 = lambda self: str(self) + '233'
z0.method01 = types.MethodType(hf0, z0)
z0.method01()

# __slots__
# @property
# @prop0.setter

#see https://docs.python.org/3/reference/datamodel.html#special-method-names
# __str__ __repr__
# __call__
# __iter__ __next__ #see https://stackoverflow.com/a/40255361/7290857
# __getitem__
# __setitem__
# __delitem__
# __getattr__ #see https://stackoverflow.com/q/4295678/7290857
# __setstate__ __getstate__ #see https://stackoverflow.com/a/41754104/7290857


# multiple inheritance, see https://docs.python.org/3/tutorial/classes.html#multiple-inheritance
# multiple inheritance passing arguments to constructors using super, see https://stackoverflow.com/q/34884567/7290857
# diamond relationship
# method resolution order MRO, see https://www.python.org/download/releases/2.3/mro/ https://zhuanlan.zhihu.com/p/43204317


class MyClass01:
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        return self.x
    def __setstate__(self, x):
        self.x = x
assert pickle.loads(pickle.dumps(MyClass01(233))).x == 233


# override operator
class MyClass02:
    def __init__(self, a):
        self.a = a
    def __add__(self, x):
        return self.a + x.a
MyClass02(2) + MyClass02(3)


# create class using type()
def hf1(self):
    print('hello from function hf1')
MyClass03 = type('MyClass03', (object,), {'hello':hf1})
print(MyClass03, type(MyClass03))
MyClass03().hello()
