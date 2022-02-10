import time
import numpy as np
import torch
import torch.utils.dlpack
import cupy as cp

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


def test_cholesky(N0=23):
    tmp0 = np.random.randn(N0,N0)
    np0 = np.matmul(tmp0, tmp0.T) + np.eye(N0)*0.1
    ret_ = np.linalg.cholesky(np0)
    ret0 = torch.cholesky(torch.tensor(np0)).numpy()
    assert hfe(ret_, ret0) < 1e-7
    if torch.cuda.is_available():
        tmp0 = torch.tensor(np0, device='cuda')
        ret1 = torch.cholesky(tmp0).to('cpu').numpy()
        assert hfe(ret_, ret1) < 1e-7


def test_cholesky_inverse(N0=23):
    tmp0 = np.random.randn(N0,N0)
    np0 = np.matmul(tmp0, tmp0.T) + np.eye(N0)*0.1
    ret_ = np.linalg.inv(np0)
    ret0 = torch.cholesky_inverse(torch.cholesky(torch.tensor(np0))).numpy()
    assert hfe(ret_, ret0) < 1e-7
    if torch.cuda.is_available():
        tmp0 = torch.tensor(np0, device='cuda')
        ret1 = torch.cholesky_inverse(torch.cholesky(tmp0)).to('cpu').numpy()
        assert hfe(ret_, ret1) < 1e-7


def test_cholesky_solve(N0=23, N1=13):
    tmp0 = np.random.randn(N0,N0)
    np0 = np.matmul(tmp0, tmp0.T) + np.eye(N0)*0.1
    np1 = np.random.randn(N0,N1)
    ret_ = np.linalg.solve(np0, np1)
    tmp0 = torch.cholesky(torch.tensor(np0))
    ret0 = torch.cholesky_solve(torch.tensor(np1), tmp0).numpy()
    assert hfe(ret_, ret0) < 1e-7
    if torch.cuda.is_available():
        tmp0 = torch.tensor(np0, device='cuda')
        tmp1 = torch.tensor(np1, device='cuda')
        ret1 = torch.cholesky_solve(tmp1, torch.cholesky(tmp0)).to('cpu').numpy()
        assert hfe(ret_, ret1) < 1e-7


def _cupy_SPD_inverse(cp0):
    '''
    solve Symmtry Positive Definite matrix inverse using Cholesky decomposition

    also see https://github.com/cupy/cupy/blob/master/cupy/linalg/solve.py
    also see https://github.com/cybertronai/pytorch-sso/blob/master/torchsso/utils/inv_cupy.py
    '''
    assert cp.cuda.cusolver_enabled
    cp0 = cp0.copy() #cp.cuda.cusolver.dpotrf() will modify in-place
    cp.linalg.util._assert_cupy_array(cp0)
    cp.linalg.util._assert_rank2(cp0)
    cp.linalg.util._assert_nd_squareness(cp0) #TODO
    assert cp0.dtype.char=='f' or cp0.dtype.char=='d'
    dtype = cp0.dtype.char

    cusolver_handle = cp.cuda.device.get_cusolver_handle()
    dev_info = cp.empty(1, dtype=cp.int)
    N0 = cp0.shape[0]
    ret = cp.eye(N0, dtype=dtype)

    if dtype == 'f':
        potrf = cp.cuda.cusolver.spotrf
        potrf_bufferSize = cp.cuda.cusolver.spotrf_bufferSize
        potrs = cp.cuda.cusolver.spotrs
    else:  # dtype == 'd'
        potrf = cp.cuda.cusolver.dpotrf
        potrf_bufferSize = cp.cuda.cusolver.dpotrf_bufferSize
        potrs = cp.cuda.cusolver.dpotrs

    buffersize = potrf_bufferSize(cusolver_handle, cp.cuda.cublas.CUBLAS_FILL_MODE_UPPER, N0, cp0.data.ptr, N0)
    workspace = cp.empty(buffersize, dtype=dtype)
    # Cholesky Decomposition
    potrf(cusolver_handle, cp.cuda.cublas.CUBLAS_FILL_MODE_UPPER, N0,
            cp0.data.ptr, N0, workspace.data.ptr, buffersize, dev_info.data.ptr)
    # solve for the inverse
    potrs(cusolver_handle, cp.cuda.cublas.CUBLAS_FILL_MODE_UPPER, N0,
            N0, cp0.data.ptr, N0, ret.data.ptr, N0, dev_info.data.ptr)
    return ret


def torch_SPD_inverse(torch0):
    # TODO support non-continuous tensor
    if torch0.is_cuda:
        tmp0 = cp.fromDlpack(torch.utils.dlpack.to_dlpack(torch0))
        ret = torch.utils.dlpack.from_dlpack(_cupy_SPD_inverse(tmp0).toDlpack())
    else:
        ret = torch.inverse(torch0) #TODO add lapack support for CPU
    return ret


def test_torch_SPD_inverse(N0=23):
    tmp0 = np.random.randn(N0, N0)
    np0 = np.matmul(tmp0, tmp0.T) + np.eye(N0)*0.1
    ret_ = np.linalg.inv(np0)

    torch0 = torch.tensor(np0)
    ret0 = torch_SPD_inverse(torch0).numpy()
    ret1 = torch_SPD_inverse(torch0.to('cuda')).to('cpu').numpy()
    assert hfe(ret_, ret0) < 1e-7
    assert hfe(ret_, ret1) < 1e-7


def demo_compare_torch_cupy_SPD_inverse_speed():
    N0 = 512
    N1 = 100
    tmp0 = np.random.randn(N0,N0)
    np0 = (np.matmul(tmp0, tmp0.T)/100 + np.eye(N0)*0.1).astype(np.float32)
    torch0 = torch.tensor(np0, device=torch.device('cuda:0'))
    t0 = time.time()
    for _ in range(N1):
        _ = torch.cholesky(torch0)
    t_torch = (time.time()-t0) / N1

    t0 = time.time()
    for _ in range(N1):
        _ = torch_SPD_inverse(torch0)
    t_cupy = (time.time()-t0) / N1
    print(f'[N0={N0}] time(torch)={t_torch}')
    print(f'[N0={N0}] time(cupy) ={t_cupy}')
