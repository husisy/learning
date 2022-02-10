def my_abs(x):
    '''
    absolute value of number

    Example:

    >>> abs(1)
    1
    >>> abs(-1)
    1
    >>> abs(0)
    0
    '''
    return x if x>=0 else (-x)


class MyDict(dict):
    '''
    simple dict that also support access as x.y style

    >>> z1 = MyDict()
    >>> z1['x'] = 100
    >>> z1.x
    100
    >>> z1.y = 233
    >>> z1['y'] = 233
    >>> z2 = MyDict(a=2, b=3, c='3')
    >>> z2.c
    '3'
    >>> z2['empty']
    Traceback (most recent call last):
        ...
    KeyError: 'empty'
    >>> z2.empty
    Traceback (most recent call last):
        ...
    AttributeError: 'MyDict' object has no attribute 'empty'
    '''
    def __init__(self, **kw):
        super(MyDict, self).__init__(**kw)
    
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError("'MyDict' object has no attribute '{}'".format(key))
    
    def __setattr__(self, key, value):
        self[key] = value
    

if __name__ == "__main__":
    import doctest
    doctest.testmod()
