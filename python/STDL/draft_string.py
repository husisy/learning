import datetime
# link: https://docs.python.org/3.7/library/string.html#format-string-syntax

'233'.format()

'{{}}/{}'.format(233) #escape

'{a}/{b}'.format(b=233, a=322) #keyword parameter

'{1}/{0}'.format(233, 322) #position parameter
'{1}/{0}/{1}'.format(233, 322)

'{0.real}/{0.imag}'.format(3+4j) #call parameter
'{0[0]}/{0[1]}'.format([23,233])

ascii('锟斤拷')
'{!a}'.format('锟斤拷')

'{:<30}'.format(233)
'{:>30}'.format(233)
'{:^30}'.format(233)
'{:*^30}'.format(233)

'{0:+f} /{0:-f} /{0: f} /{0:f}'.format(2.33)

'int: {0:d}; hex: {0:x}; oct: {0:o}; bin: {0:b}'.format(42)

'{:,}'.format(1234567890)

'{:.2%}'.format(0.234567)

x1 = datetime.datetime(2010,7,3,14,15,58)
'{:%Y-%m-%d %H:%M:%S}'.format(x1)

class MyClass1(object):
    def __init__(self, num1):
        self.num1 = num1
    def __str__(self):
        return str(self.num1) + '233__str__()'
    def __repr__(self):
        return str(self.num1) + '锟斤拷__repr__()'

x1 = MyClass1(233)
'{0.num1}'.format(x1)
'{!s}'.format(x1)
'{!r}'.format(x1)
'{!a}'.format(x1)

z0 = "z0-abb"
f'z0 is {z0}'
f'z0.upper() is {z0.upper()}'
