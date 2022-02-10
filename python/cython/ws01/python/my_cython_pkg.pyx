from libc.stdlib cimport atoi
from libc.math cimport sin


def hello(name):
    print('hello {}'.format(name))


cdef double _demo_integrate_hf0(double x)except? -2:
    return x ** 2 - x

def demo_integrate(double a, double b, int N):
    cdef int i
    cdef double s
    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += _demo_integrate_hf0(a + i * dx)
    return s * dx


def primes(unsigned int nb_primes):
    cdef int n, i, len_p
    cdef int p[1000]
    if nb_primes > 1000:
        nb_primes = 1000
    len_p = 0
    n = 2
    while len_p < nb_primes:
        for i in p[:len_p]:
            if n % i == 0:
                break
        else:
            p[len_p] = n
            len_p += 1
        n += 1
    ret = [x for x in p[:len_p]]
    return ret


def demo_libc_stdlib_atoi(char* s):
    assert s is not NULL, 'byte string value is NULL'
    ret = atoi(s)
    return ret


def demo_libc_math_sin(double x):
    ret = sin(x)
    return ret
