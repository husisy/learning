import numpy as np

import my_f2py_module


np_rng = np.random.default_rng()
hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
hf_randc = lambda *size: (np_rng.normal(size=size) + 1j*np_rng.normal(size=size)).astype(np.complex128)


def test_zadd():
    N0 = 23
    np0 = hf_randc(N0)
    np1 = hf_randc(N0)
    ret_ = np0 + np1
    ret0 = np.zeros_like(np0)
    my_f2py_module.zadd(np0, np1, ret0, N0)
    assert hfe(ret_, ret0) < 1e-5


def test_fib():
    hf0 = lambda x: x if x < 2 else hf0(x-1) + hf0(x-2)
    num0 = 8
    ret0 = np.zeros(num0, dtype=np.float64)
    my_f2py_module.fib(ret0, num0)
    ret1 = np.array([hf0(x) for x in range(num0)])
    assert np.abs(ret0-ret1).max() < 1e-7
