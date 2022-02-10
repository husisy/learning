from . import _cpp

def pyadd(a, b):
    return _cpp.add(a, b) #defined in src/main.cpp
