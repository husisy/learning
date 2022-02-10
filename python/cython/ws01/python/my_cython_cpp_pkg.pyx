# distutils: language=c++
from libcpp.vector cimport vector

def prime_cvector(unsigned int nb_primes):
    cdef int n, i
    cdef vector[int] p
    n = 2
    while p.size() < nb_primes:
        for i in p:
            if n % i == 0:
                break
        else:
            p.push_back(n)
        n += 1
    return p
