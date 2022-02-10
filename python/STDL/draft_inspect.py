import inspect


hf0 = lambda a,b=233: a+b
z0 = inspect.signature(hf0)
for x,y in z0.parameters.items():
    print(x, y.default)
# a <class 'inspect._empty'>
# b 233


class Dummy0:
    pass
class Dummy1(Dummy0):
    pass
z0 = inspect.getmro(Dummy1) #(tuple)
# (__main__.Dummy1, __main__.Dummy0, object)
