import numpy as np
import scipy.integrate

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

def test_basic():
    xmin = 0
    xmax = 4
    hf0 = lambda x: x**2
    ret_ = 4**3/3
    ret0,err_ = scipy.integrate.quad(hf0, xmin, xmax)
    assert hfe(ret_,ret0) < 1e-7
    assert err_  < 1e-7

    xmin = 0
    xmax = np.inf
    hf0 = lambda x: np.exp(-x)
    ret_ = 1
    ret0,err_ = scipy.integrate.quad(hf0, xmin, xmax)
    assert hfe(ret_,ret0) < 1e-7
    assert err_ < 1e-7
