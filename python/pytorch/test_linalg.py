import numpy as np
import torch
import scipy.linalg
import scipy.sparse.linalg

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

def test_torch_inverse(N0=3, N1=5):
    np0 = np.random.rand(N0, N1, N1) + np.eye(N1)*N1/2
    ret_ = np.stack([np.linalg.inv(x) for x in np0])
    torch0 = torch.tensor(np0)
    ret0 = torch.inverse(torch0)
    assert hfe(ret_, ret0.numpy()) < 1e-5


def test_torch_svd(N0=3, N1=23):
    np0 = np.random.randn(N0, N1, N1)
    ret_ = np.stack([np.linalg.svd(x, compute_uv=False) for x in np0])
    ret_U, ret_S, ret_V = torch.svd(torch.tensor(np0))
    ret_U, ret_S, ret_V = ret_U.numpy(), ret_S.numpy(), ret_V.numpy()
    assert hfe(ret_, ret_S) < 1e-7
    assert max(hfe(np.eye(N1), x@x.T) for x in ret_U) < 1e-7
    assert max(hfe(np.eye(N1), x@x.T) for x in ret_V) < 1e-7
    assert hfe(np0, np.stack([(x*y)@z.T for x,y,z in zip(ret_U,ret_S,ret_V)])) < 1e-7


def test_torch_symeig(N0=3, N1=23):
    tmp0 = np.random.rand(N0, N1, N1)
    np0 = tmp0 + tmp0.transpose(0,2,1)
    ret_ = np.stack([np.linalg.eigvalsh(x) for x in np0])
    EVL,EVC = torch.symeig(torch.tensor(np0), eigenvectors=True)
    EVL, EVC = EVL.numpy(), EVC.numpy()
    assert hfe(ret_, EVL) < 1e-7
    assert max(hfe(np.eye(N1), x@x.T) for x in EVC) < 1e-7
    assert hfe(np0, np.stack([(y*x)@y.T for x,y in zip(EVL,EVC)])) < 1e-7


def rand_hermite_matrix(N0, min_eig=1, max_eig=2, tag_complex=True, seed=None):
    rand_generator = np.random.RandomState(seed)
    ret = rand_generator.randn(N0, N0)
    if tag_complex:
        ret = ret + 1j*rand_generator.randn(N0,N0)
    ret = ret + np.conjugate(ret.T)
    eig0 = scipy.sparse.linalg.eigsh(ret, k=1, which='SA', return_eigenvectors=False)
    eig1 = scipy.sparse.linalg.eigsh(ret, k=1, which='LA', return_eigenvectors=False)
    ret = (ret - eig0*np.eye(N0)) * (max_eig-min_eig)/(eig1-eig0) + min_eig*np.eye(N0)
    return ret


def test_fractional_matrix_power(N0=5):
    np0 = rand_hermite_matrix(N0, tag_complex=False)
    np1 = np.random.rand()
    ret_ = scipy.linalg.fractional_matrix_power(np0, np1)
    EVL,EVC = torch.symeig(torch.tensor(np0), eigenvectors=True)
    ret0 = torch.matmul(EVC * (EVL**np1), EVC.t_()).numpy()
    assert hfe(ret_, ret0) < 1e-7


def test_linalg_det(N0=3, N1=23):
    np0 = np.random.rand(N0, N1, N1).astype(np.float32)
    ret_ = np.linalg.det(np0)

    torch0 = torch.tensor(np0)
    ret0 = torch.linalg.det(torch0).numpy()
    tmp0,tmp1 = torch.linalg.slogdet(torch0)
    ret1 = (tmp0 * torch.exp(tmp1)).numpy()
    assert hfe(ret_, ret0) < 1e-5
    assert hfe(ret_, ret1) < 1e-5
