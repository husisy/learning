import numpy as np
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
hfH = lambda x: x + np.conjugate(x.T)
hf_randc = lambda *size: np.random.randn(*size) + 1j*np.random.randn(*size)

# see README.md/sparse for usage guide

def test_csr_matrix():
    snp0 = scipy.sparse.csr_matrix([[1,2,0],[0,0,3],[4,0,5]])
    tmp0 = np.array([1,0,-1])
    tmp1 = np.dot(np.array([[1,2,0],[0,0,3],[4,0,5]]), tmp0)
    assert hfe(snp0.dot(tmp0), tmp1) < 1e-7


def test_coo_matrix():
    I = np.array([0,3,1,0])
    J = np.array([0,3,1,2])
    V = np.array([4,5,7,9])
    A = scipy.sparse.coo_matrix((V,(I,J)),shape=(4,4))
    np0 = np.array([[4,0,9,0],[0,7,0,0],[0,0,0,0],[0,0,0,5]], dtype=np.int64)
    assert hfe(A.todense(), np0) < 1e-7


def test_sparse_linalg_expm_multiply(N0=3, start=1, stop=2, N1=5):
    np0 = hfH(hf_randc(N0,N0))
    np1 = hf_randc(N0)
    ret_ = np.stack([np.dot(scipy.linalg.expm(np0*x), np1) for x in np.linspace(start,stop,N1)])
    ret0 = scipy.sparse.linalg.expm_multiply(np0, np1, start=start, stop=stop, num=N1, endpoint=True)
    assert hfe(ret_,ret0) < 1e-7


def test_sparse_hermite():
    N0 = 100
    z0 = scipy.sparse.rand(N0, N0, density=0.1, format='csr')
    z1 = z0 + z0.T
    hf_sparse_is_hermite = lambda x,eps=1e-5: ((abs(x-x.T) >= eps).nnz)==0
    assert not hf_sparse_is_hermite(z0)
    assert hf_sparse_is_hermite(z1)
