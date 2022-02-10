import numpy as np
import cupy as cp

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
assert cp.cuda.is_available()


def test_cusolver_cholesky(N0=23):
    tmp0 = np.random.rand(N0,N0)
    np0 = np.matmul(tmp0, tmp0.T) + np.eye(N0)*0.1
    ret_ = np.linalg.cholesky(np0)

    cp0 = cp.array(np0)
    cusolver_handle = cp.cuda.device.get_cusolver_handle()
    dev_info = cp.empty(1, dtype=cp.int)
    buffersize = cp.cuda.cusolver.dpotrf_bufferSize(cusolver_handle, cp.cuda.cublas.CUBLAS_FILL_MODE_UPPER, N0, cp0.data.ptr, N0)
    workspace = cp.empty(buffersize, dtype=cp0.dtype)
    # see LAPACK/modules/positive-definite-matrix/computational-routines-double/dpotrf http://www.netlib.org/lapack/explore-html/index.html
    cp.cuda.cusolver.dpotrf(cusolver_handle, cp.cuda.cublas.CUBLAS_FILL_MODE_UPPER, N0,
            cp0.data.ptr, N0, workspace.data.ptr, buffersize, dev_info.data.ptr)
    ret0 = cp.tril(cp0).get()
    assert hfe(ret_, ret0) < 1e-7


def test_cusolver_dpotrs_SDP_solve(N0=23):
    np0 = np.tril(np.random.rand(N0,N0)) + np.eye(N0)*0.1
    ret_ = np.linalg.inv(np.matmul(np0, np0.T))

    # the upper triangle matrix element is useless
    cusolver_handle = cp.cuda.device.get_cusolver_handle()
    dev_info = cp.empty(1, dtype=cp.int)
    cp0 = cp.array(np0)
    cp0 = cp0 + cp.triu(cp.random.normal(size=(N0,N0),dtype=cp0.dtype),k=1)
    ret0 = cp.eye(N0, dtype=cp0.dtype)
    # see LAPACK/modules/positive-definite-matrix/computational-routines-double/dpotrs http://www.netlib.org/lapack/explore-html/index.html
    cp.cuda.cusolver.dpotrs(cusolver_handle, cp.cuda.cublas.CUBLAS_FILL_MODE_UPPER, N0,
            N0, cp0.data.ptr, N0, ret0.data.ptr, N0, dev_info.data.ptr)
    assert hfe(ret_, ret0.get()) < 1e-7


def test_cublas_dot():
    n = 3
    cp0 = cp.random.randn(3)
    cp1 = cp.random.randn(3)
    cp2 = cp.empty(1, dtype=cp.float64)
    ret_ = cp.dot(cp0, cp1).get()
    cublas_handle = cp.cuda.device.get_cublas_handle()
    cp.cuda.cublas.ddot(cublas_handle, n, cp0.data.ptr, 1, cp1.data.ptr, 1, cp2.data.ptr)
    ret0 = cp2.get()
    assert abs(ret_-ret0) < 1e-10
