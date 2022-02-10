# -*- coding: utf-8 -*-
import math

## help
help(print)
help(type)
help(help)
# ?help # in ipython


## io
# x0 = input("what's your ni da ye name: ")
print('233')
print(233)
print(233, '233')
print(True)


## string
type('233')
'233'
'\''
"\""
'\\'
"233"
'''233
233
233'''
'233: {}'.format(233)
'233' + '33'
'2' * 3
chr(65)
ord('a')


## type convert
type(233)
type(233.0)
str(233)
str(233.0)
int('233')
float('233.0')
float(233)
233.2//1
math.ceil(232.2)
round(232.7)
round(233.2)
math.floor(233.7)


## boolean
type(True)
True and False
True or False
bool(1)
bool('')
all(x for x in [True,True])
all(x for x in [])
any(x for x in [True,False])
any(x for x in [])


## list
type([2,3,3])
x = [2,3,3]
x.append(233)
x[0]
x[-1]
x[:-1]
x[1:]
x[::-1]
len(x)
x = [y*y for y in range(11) if y%2==0]
x = [y0+y1 for y0 in 'ABC' for y1 in 'XYZ']


# slice
x = slice(0,3)
y = [2,3,4,5,6]
y[x]
slice(None, None, None)
slice(None, None, -1)


## tuple
x = (2,3,3)
x = 2, 3, 3
x,y,z = 2,3,3
x = (y for y in [2,3,3])
tuple(x)
tuple([2,3,3])


## dict
x = {'m':1,'d':2,'z':3,'Z':4}
x = dict(m=1, d=2, z=3, Z=4)
x['m']
x.get('M', -1)
'M' in x


## set
x = {2,3,3}
x = set([2,3,3])
{2,3,3} & {3,4}
{2,3,3} | {3,4}
{2,3,3} - {3,4}
'233' in {2,3,3} #False


## NoneType
None
None is None #True
None == None #False


## logical structure
if 2 > 3:
    print('233')
else:
    print('2333')
x = 233 if 2>3 else 2333

x = 0
for y in range(1, 101):
    x += y
x = sum(y for y in range(1, 101))
x = 0
y = 1
while y < 101:
    x += y
    y += 1


## misc
assert True
if True:
    pass


## function
abs(-1)
range(233, 0, -1)
math.sin(math.pi)
math.cos(math.pi)
sorted([3,2,3])
len('233')


## type and instance
type('233')
isinstance('233', str)
type(233)
isinstance(233, int)
isinstance(233, (int,float))
