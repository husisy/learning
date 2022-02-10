# cython: infer_types=True
# distutils: extra_compile_args=-fopenmp
# distutils: extra_link_args=-fopenmp
cimport cython
import cython.parallel
import numpy as np

cdef int clip00(int a, int min_value, int max_value):
    return min(max(a, min_value), max_value)


@cython.boundscheck(False) # Deactivate bounds checking
@cython.wraparound(False) # Deactivate negative indexing
def hf_dummy00(int[:,:] np0, int[:,:] np1, int a, int b, int c):

    cdef Py_ssize_t num0 = np0.shape[0]
    cdef Py_ssize_t num1 = np0.shape[1]

    assert tuple(np0.shape) == tuple(np1.shape)

    ret = np.zeros((num0, num1), dtype=np.intc)
    cdef int[:,:] ret_view = ret
    cdef Py_ssize_t x, y

    for x in range(num0):
        for y in range(num1):
            ret_view[x, y] = clip00(np0[x, y], 2, 10) * a + np1[x, y] * b + c

    return ret


ctypedef fused my_type:
    int
    long long
    float
    double

cdef my_type clip01(my_type a, my_type min_value, my_type max_value) nogil:
    return min(max(a, min_value), max_value)

@cython.boundscheck(False)
@cython.wraparound(False)
def hf_dummy01(my_type[:,:] np0, my_type[:,:] np1, my_type a, my_type b, my_type c):

    num0 = np0.shape[0]
    num1 = np0.shape[1]

    assert tuple(np0.shape) == tuple(np1.shape)

    if my_type is int:
        dtype = np.int32
    elif my_type is cython.longlong:
        dtype = np.int64
    elif my_type is float:
        dtype = np.float32
    elif my_type is double:
        dtype = np.float64

    result = np.zeros((num0, num1), dtype=dtype)
    cdef my_type[:, ::1] result_view = result
    cdef Py_ssize_t x, y

    # for x in range(num0):
    for x in cython.parallel.prange(num0, nogil=True):
        for y in range(num1):
            result_view[x, y] = clip01(np0[x, y], 2, 10) * a + np1[x, y] * b + c
    return result
