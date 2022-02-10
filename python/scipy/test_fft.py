import numpy as np
import scipy.fft

hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))

def test_fft(N0=3, N1=5):
    hf_fft_mat = lambda N0: np.exp(-2j*np.pi/N0*(np.arange(N0)[:,np.newaxis]*np.arange(N0)))
    np0 = np.random.randn(N0, N1) + np.random.randn(N0,N1)*1j

    ret_ = np0 @ hf_fft_mat(N1)
    ret0 = np.fft.fft(np0, axis=1)
    assert hfe(ret_, ret0) < 1e-7

    ret_ = hf_fft_mat(N0) @ np0 @ hf_fft_mat(N1)
    ret0 = np.fft.fftn(np0, axes=(0,1))
    assert hfe(ret_, ret0) < 1e-7


def test_ifft(N0=3, N1=5):
    hf_ifft_mat = lambda N0: np.exp(2j*np.pi/N0*(np.arange(N0)[:,np.newaxis]*np.arange(N0)))/N0
    np0 = np.random.randn(N0, N1) + np.random.randn(N0,N1)*1j

    ret_ = np0 @ hf_ifft_mat(N1)
    ret0 = np.fft.ifft(np0, axis=1)
    assert hfe(ret_, ret0) < 1e-7

    ret_ = hf_ifft_mat(N0) @ np0 @ hf_ifft_mat(N1)
    ret0 = np.fft.ifftn(np0, axes=(0,1))
    assert hfe(ret_, ret0) < 1e-7


def test_fftshift():
    N0 = 10
    np0 = np.random.rand(N0)
    ret_ = np.concatenate([np0[((N0+1)//2):],np0[:((N0+1)//2)]])
    ret0 = np.fft.fftshift(np0)
    assert hfe(ret_, ret0) < 1e-7

    N0 = 11
    np0 = np.random.rand(N0)
    ret_ = np.concatenate([np0[((N0+1)//2):],np0[:((N0+1)//2)]])
    ret0 = np.fft.fftshift(np0)
    assert hfe(ret_, ret0) < 1e-7

def test_fft_dst():
    N0 = 32
    np0 = np.random.rand(N0)
    np1 = scipy.fft.dst(np0, type=1)
    np1_ = 2*np.sin(np.pi*np.arange(1,N0+1)[:,np.newaxis]*np.arange(1,N0+1)/(N0+1)) @ np0
    np2 = scipy.fft.idst(np1, type=1)
    np2_ = 1/(N0+1)*np.sin(np.pi*np.arange(1,N0+1)[:,np.newaxis]*np.arange(1,N0+1)/(N0+1)) @ np1_
    assert hfe(np1,np1_)<1e-7
    assert hfe(np0,np2_)<1e-7
