import numpy as np
import cupy as cp

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

assert cp.cuda.is_available()


x = cp.array([1,2,3])
x.dtype
x.shape
x.device

with cp.cuda.Device(0):
    x = cp.random.rand(4)
    y = cp.linalg.norm(x) #x.device MUST be the same as the currentt device


cp.cuda.Device(0).use()
cp.cuda.get_device_id()


np0 = np.array([1j,2,3j], dtype=np.complex128)
np1 = np0.view(np.float64)
np2 = np1.view(np.complex128)


cp0 = cp.array([1j,2,3j], dtype=cp.complex128)
cp1 = cp0.view(dtype=cp.float64)
cp2 = cp1.view(dtype=cp.complex128)
