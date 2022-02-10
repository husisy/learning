import numpy as np
import scipy.integrate
import numba

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
np_rng = np.random.default_rng()

@numba.cfunc('float64(float64, float64)')
def hf0_cfunc(x, y):
    return x + y
hf0_cfunc.ctypes(4.0, 5.0)
hf0_cfunc.address #for call from c/cpp library


hf0 = lambda x: np.exp(-x)/(x**2)
hf1_cfunc = numba.cfunc('float64(float64)')(hf0)
ret0 = scipy.integrate.quad(hf0, 1, np.inf)[0]
ret1 = scipy.integrate.quad(hf1_cfunc.ctypes, 1, np.inf)[0] #not invoke the python interpreter
assert hfe(ret0,ret1) < 1e-5


from numba import cfunc, types, carray

tmp0 = numba.types.void(numba.types.CPointer(numba.types.double),
        numba.types.CPointer(numba.types.double), numba.types.intc, numba.types.intc)
@numba.cfunc(tmp0)
def my_callback(in_, out, m, n):
    in_array = numba.carray(in_, (m, n)) #c-order, otherwise numba.farray for fortran-order
    out_array = numba.carray(out, (m, n))
    for i in range(m):
        for j in range(n):
            out_array[i, j] = 2 * in_array[i, j]
# np0 = np_rng.uniform(0, 1, size=(2,3)).astype(np.float64)
# np1 = np.zeros_like(np0)
# my_callback.ctypes(np0.ctypes, np1.ctypes, *np0.shape) #TODO
