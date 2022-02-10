import numpy as np

import my_cython_pkg

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

def hf_dummy00(np0, np1, a, b, c):
    ret = np.clip(np0, 2, 10) * a + np1 * b + c
    return ret


np0 = np.random.uniform(0, 1000, size=(3000, 2000)).astype(np.int32)
np1 = np.random.uniform(0, 1000, size=(3000, 2000)).astype(np.int32)
a = 4
b = 3
c = 9

ret_ = hf_dummy00(np0, np1, a, b, c)
# %timeit ret_ = hf_dummy00(np0, np1, a, b, c) #23.1ms

ret0 = my_cython_pkg.hf_dummy00(np0, np1, a, b, c)
# %timeit ret0 = my_cython_pkg.hf_dummy00(np0, np1, a, b, c) #13.9ms


for dtype in [np.int32, np.int64, np.float32, np.float64]:
    _ = my_cython_pkg.hf_dummy01(np0.astype(dtype), np1.astype(dtype), a, b, c)
