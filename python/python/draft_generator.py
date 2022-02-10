def hf1():
    print('line-0')
    x = yield '0'

    print('line-1')
    print('x={}'.format(x))
    x = yield '1'

    print('line-2')
    print('x={}'.format(x))

c = hf1()
y = c.send(None)
# print('line-0')
# y will be '0'

y = c.send('a')
# x will be 'a'
# print('line-1')
# y will bee '1'

c.send('b')
# x will be 'b'
# print('line-2')
# cannot assign y (StopIteration excetpion)

c.close()
