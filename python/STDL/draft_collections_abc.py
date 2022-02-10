from collections.abc import Generator, Iterable, Iterator


## generator
def math_fib(max_iteration):
    n,a,b = 0, 0, 1
    while n<max_iteration:
        yield b
        a,b = b,a+b
        n += 1
    return 'done'
for x in math_fib(5):
    print(x)
isinstance(math_fib(3), Generator)

def hf1():
    print('start after send None')
    x = yield 'A'
    print('x is "{}"'.format(x))
    x = yield 'B'
    print('x is "{}"'.format(x))
c = hf1()
y = c.send(None)
print('y is "{}"'.format(y))
# start after send None
# y is "A"
y = c.send('a')
print('y is "{}"'.format(y))
# x is "a"
# y is "B"
try:
    c.send('b')
except StopIteration as e:
    y = e.value
print('y is "{}"'.format(y))
# x is "b"
# y is "None"
c.close()


## Iterable and Iterator
def hf2():
    yield 233
z1 = [
    '233',
    [2,3,3],
    (2,3,3),
    iter('233'),
    hf2(),
]
str_format = '{}\t{}\t{}\t{}'
print(str_format.format('id', 'type', 'isIterable', 'isIterator'))
for x in z1:
    print(str_format.format(x, type(x), isinstance(x,Iterable), isinstance(x, Iterator)))
