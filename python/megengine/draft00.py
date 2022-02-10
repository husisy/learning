import numpy as np
import megengine as mge
# import megengine.functional as F
import megengine.module as M

hfe = lambda x,y,eps=1e-3: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


# megengine not support float64 yet
x0 = mge.tensor(np.random.rand(2, 5).astype(np.float32))
x1 = mge.tensor([2,23,233])
x0.set_value(np.random.rand(2,5))
x0.dtype #np.float32 np.float16 np.int32
x0.shape
x0.reshape(5,2)
x0.numpy()
x0.device


# indexing
x0 = mge.tensor(np.random.rand(2, 5).astype(np.float32))
x0[:,0]
x0[0,0]


# operator
np0 = np.random.rand(3,4).astype(np.float32)
np1 = np.random.rand(4,3).astype(np.float32)
ret_ = np0 @ np1
ret0 = mge.tensor(np0) @ mge.tensor(np1)
ret1 = mge.functional.matrix_mul(mge.tensor(np0), mge.tensor(np1))
assert hfe(ret_, ret0.numpy()) < 1e-4
assert hfe(ret_, ret1.numpy()) < 1e-4


# device
x0 = mge.tensor(np.random.rand(2, 5).astype(np.float32))
x1 = x0.to('cpu0')
x0.device #gpu0
x1.device #cpu0


# autograd
x0 = mge.tensor(np.random.rand(2, 5).astype(np.float32))
x1 = (x0**2).sum()
grad_x0 = mge.functional.grad(x1, x0, use_virtual_grad=False)


# parameter
N0 = 23
N1 = 3
N2 = 5
np0 = np.random.randn(N0, N1).astype(np.float32)
np_weight = np.random.randn(N1,N2).astype(np.float32)
np_bias = np.random.randn(N2).astype(np.float32)
np1 = (np0 @ np_weight) + np_bias
mge0 = mge.tensor(np0)
mge_weight = mge.Parameter(np_weight.T)
mge_bias = mge.Parameter(np_bias)
mge1 = mge.functional.linear(mge0, mge_weight, mge_bias)
assert hfe(np1, mge1.numpy()) < 1e-4


## module
