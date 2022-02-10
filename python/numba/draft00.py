import time
import numba
import numpy as np

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

def my_measure_time(hf0, num_repeat=10):
    def _hf_time():
        t0 = time.time()
        hf0()
        ret = time.time() - t0
        return ret
    compile_time = _hf_time()
    tmp0 = np.array([_hf_time() for _ in range(num_repeat)])
    mean_time = tmp0.mean()
    std_time = np.std(tmp0)
    return mean_time, std_time, compile_time

@numba.jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def hf0_numba(a): # Function is compiled to machine code when called the first time
    trace = 0.0
    for i in range(a.shape[0]):
        trace += np.tanh(a[i, i])
    ret = a + trace
    return ret


def hf0_np_vectorization(a):
    ret = a + np.tanh(np.diag(a)).sum()
    return ret
# x = np.arange(100).reshape(10, 10)
# z0 = go_fast(x)

np0 = np.random.randn(1024, 1024)
print(my_measure_time(lambda: hf0_numba(np0), num_repeat=100))
print(my_measure_time(lambda: hf0_np_vectorization(np0), num_repeat=100))



# numpy.ufunc
tmp0 = [
    numba.int32(numba.int32, numba.int32),
    numba.int64(numba.int64, numba.int64),
    numba.float32(numba.float32, numba.float32),
    numba.float64(numba.float64, numba.float64)
]
@numba.vectorize(tmp0, nopython=True)
def hf_numba_add(x, y):
    return x + y
hf_numba_add(np.random.uniform(0, 1, size=(2,3)).astype(np.float32), np.random.uniform(0, 1, size=(2,3)).astype(np.float32))
hf_numba_add(np.random.uniform(0, 1, size=(2,3)).astype(np.float64), np.random.uniform(0, 1, size=(2,3)).astype(np.float64))
hf_numba_add.reduce(np.arange(12).reshape(3,4), axis=1)
hf_numba_add.accumulate(np.arange(12).reshape(3,4), axis=1)


@numba.guvectorize([(numba.int64[:], numba.int64, numba.int64[:])], '(n),()->(n)', nopython=True)
def hf00_guvectorize(x, y, ret0):
    for i in range(x.shape[0]):
        ret0[i] = x[i] + y
hf00_guvectorize(np.arange(5), 2)
hf00_guvectorize(np.arange(6).reshape(2,3), 2)


@numba.guvectorize([(numba.int64[:], numba.int64, numba.int64[:], numba.int64[:])], '(n),()->(n),(n)', nopython=True)
def hf01_guvectorize(x, y, ret0, ret1):
    for i in range(x.shape[0]):
        ret0[i] = x[i] + y
        ret1[i] = x[i] - y
hf01_guvectorize(np.arange(5), 2)
hf01_guvectorize(np.arange(6).reshape(2,3), 2)


## dynamic universal functions
@numba.vectorize(nopython=True)
def hf00_dynamic_vectorize(x, y):
    ret = x*y
    return ret
hf00_dynamic_vectorize.ufunc
hf00_dynamic_vectorize.ufunc.types #[]
hf00_dynamic_vectorize(3, 4)
hf00_dynamic_vectorize(3.0, 4.0)
hf00_dynamic_vectorize(3.0, 4) #not create new kernel



@numba.guvectorize('(n),()->(n)')
def hf00_dynamic_guvectorize(x, y, res):
    for i in range(x.shape[0]):
        res[i] = x[i] + y
x = np.arange(5, dtype=np.int64)
ret = np.zeros_like(x)
hf00_dynamic_guvectorize(x, 23, ret) #ret must be given in dynamic numba.guvectorize case

x = np.arange(5, dtype=np.float64)
ret = np.zeros_like(x)
hf00_dynamic_guvectorize(x, 2.3, ret)

x = np.arange(5, dtype=np.float64)
ret = np.zeros_like(x)
hf00_dynamic_guvectorize(x, 23, ret) #not create new kernel

x = np.arange(5, dtype=np.float64)
ret0 = np.zeros(5, dtype=np.int64)
hf00_dynamic_guvectorize(x, 2.3, ret0) #create new kernel for 'dd->l'


## jitclass
tmp0 = [
    ('value', numba.int32),
    ('array', numba.float32[:]),
]
@numba.experimental.jitclass(tmp0)
class MyDummyJITClass00:
    def __init__(self, value):
        self.value = value
        self.array = np.zeros(value, dtype=np.float32)

    @property
    def size(self):
        return self.array.size

    def increment(self, val):
        for i in range(self.size):
            self.array[i] += val
        return self.array

    @staticmethod
    def add(x, y):
        return x + y
z0 = MyDummyJITClass00(23)
z0.size
z0.value
z0.array
z0.increment(23)
MyDummyJITClass00.add(2, 23)
