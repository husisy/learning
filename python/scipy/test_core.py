import numpy as np
import scipy

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

def test_poly1d(N0=3, N1=5):
    np_coefficient = np.random.rand(N0) #from high order to low
    np_root = np.random.rand(N0)
    np_x = np.random.rand(N1)
    ret_ = np.sum(np_coefficient[::-1,np.newaxis] * (np_x**(np.arange(N0))[:,np.newaxis]), axis=0)
    ret0 = np.poly1d(np_coefficient)(np_x)
    assert hfe(ret_, ret0) < 1e-7

    ret_ = np.product(np_x-np_root[:,np.newaxis], axis=0)
    ret0 = np.poly1d(np_root, r=True)(np_x)
    assert hfe(ret_, ret0) < 1e-7


def test_pad(N0=3, N1=5):
    np0 = np.random.rand(N0, N1)

    ret0 = np.zeros((N0+2,N1+2), dtype=np0.dtype)
    ret0[0,1:-1] = np0[-1]
    ret0[-1,1:-1] = np0[0]
    ret0[1:-1,0] = np0[:,-1]
    ret0[1:-1,-1] = np0[:,0]
    ret0[0,0] = np0[-1,-1]
    ret0[0,-1] = np0[-1,0]
    ret0[-1,0] = np0[0,-1]
    ret0[-1,-1] = np0[0,0]
    ret0[1:-1,1:-1] = np0
    ret_ = np.pad(np0, [(1,1),(1,1)], mode='wrap')
    assert hfe(ret_, ret0) < 1e-7


def test_complex_abs():
    N0 = 23
    np0 = (np.random.randn(N0) + 1j*np.random.randn(N0)).astype(np.complex128)
    ret_ = np.abs(np0)**2
    tmp0 = np0.view(np.float64)
    ret0 = np.matmul(tmp0.reshape(N0,1,2), tmp0.reshape(N0,2,1)).reshape(N0)
    assert hfe(ret_,ret0) < 1e-7


def test_complex_vdot():
    hf_randc = lambda *size: np.random.randn(*size) + 1j*np.random.randn(*size)
    np0 = hf_randc(3, 4)
    np1 = hf_randc(3, 4)
    ret_ = np.dot(np0.conj().reshape(-1), np1.reshape(-1))
    ret0 = np.vdot(np0.reshape(-1), np1.reshape(-1))
    assert np.abs(ret_-ret0) < 1e-7
