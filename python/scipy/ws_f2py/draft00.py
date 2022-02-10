import add
import numpy as np

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
hf_randc = lambda *x: np.random.randn(*x) + 1j*np.random.randn(*x)

N0 = 23
np0 = hf_randc(N0)
np1 = hf_randc(N0)
ret_ = np0 + np1
ret0 = np.zeros_like(np0)
add.zadd(np0, np1, ret0, N0)
assert hfe(ret_, ret0) < 1e-5
