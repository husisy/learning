import numpy as np
import torch

# also see python/scipy/test_fft.py

hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
np_rng = np.random.default_rng()
hf_randc = lambda *size: np_rng.normal(size=size) + 1j*np_rng.normal(size=size)
hf_fft_mat = lambda N0: np.exp(-2j*np.pi/N0*(np.arange(N0)[:,np.newaxis]*np.arange(N0)))
hf_ifft_mat = lambda N0: np.exp(2j*np.pi/N0*(np.arange(N0)[:,np.newaxis]*np.arange(N0)))/N0

def test_fft():
    N0 = 3
    N1 = 5
    np0 = hf_randc(N0, N1)
    torch0 = torch.tensor(np0, dtype=torch.complex128)

    ret_ = np0 @ hf_fft_mat(N1)
    ret0 = torch.fft.fft(torch0, dim=1).numpy()
    assert hfe(ret_, ret0) < 1e-7

    ret_ = hf_fft_mat(N0) @ np0 @ hf_fft_mat(N1)
    ret0 = torch.fft.fftn(torch0, dim=(0,1)).numpy()
    assert hfe(ret_, ret0) < 1e-7


def test_ifft():
    N0 = 3
    N1 = 5
    np0 = hf_randc(N0, N1)
    torch0 = torch.tensor(np0, dtype=torch.complex128)

    ret_ = np0 @ hf_ifft_mat(N1)
    ret0 = torch.fft.ifft(torch0, dim=1).numpy()
    assert hfe(ret_, ret0) < 1e-7

    ret_ = hf_ifft_mat(N0) @ np0 @ hf_ifft_mat(N1)
    ret0 = torch.fft.ifftn(torch0, axis=(0,1)).numpy()
    assert hfe(ret_, ret0) < 1e-7


def test_fftshift():
    N0 = 10
    np0 = np.random.rand(N0)
    torch0 = torch.tensor(np0, dtype=torch.float64)
    ret_ = np.concatenate([np0[((N0+1)//2):],np0[:((N0+1)//2)]])
    ret0 = torch.fft.fftshift(torch0).numpy()
    assert hfe(ret_, ret0) < 1e-7

    N0 = 11
    np0 = np.random.rand(N0)
    torch0 = torch.tensor(np0, dtype=torch.float64)
    ret_ = np.concatenate([np0[((N0+1)//2):],np0[:((N0+1)//2)]])
    ret0 = torch.fft.fftshift(torch0).numpy()
    assert hfe(ret_, ret0) < 1e-7
