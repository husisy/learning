import pyfftw
import numpy as np

hfe = lambda x,y:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))

N0 = 128
np0 = (np.random.randn(N0) + 1j*np.random.randn(N0)).astype(np.complex128)
ret_ = np.fft.fft(np0)
tmp0 = pyfftw.empty_aligned(N0, dtype='complex128', n=16)
tmp0[:] = np0
ret0 = pyfftw.interfaces.numpy_fft.fft(tmp0)
assert hfe(ret_, ret0) < 1e-7
