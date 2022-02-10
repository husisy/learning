from collections import namedtuple, deque, defaultdict, ChainMap


MyNamedTuple = namedtuple('MyNamedTuple', ['x','y'])
x0 = MyNamedTuple(2,3)
x0.x, x0[0]
x0.y, x0[1]
# x0[0] = 233 #fail, tuple is immutable
isinstance(x0, MyNamedTuple) #True
isinstance(x0, tuple) #True


x1 = deque('abc')
x1.append('x') #abcx
x1.appendleft('y') #yabcx
x1.pop() #yabc x
x1.popleft() #abc y


x1 = defaultdict()
x1 = defaultdict(list)
x1 = defaultdict(lambda: 'None')


x1 = {'a':'a_1', 'b':'b_1'}
x2 = {'b':'b_2', 'c':'c_2'}
z1 = ChainMap(x1,x2)
z1['a'], z1['b'], z1['c']
list(z1.items())