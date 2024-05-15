import numpy as np
import scipy.sparse
import torch
# https://pytorch.org/docs/stable/sparse.html
# https://github.com/pydata/sparse

np_rng = np.random.default_rng()
hf_randc = lambda *x: np_rng.normal(size=x) + 1j*np_rng.normal(size=x)

# 2 dimension (coo default)
N0,N1,N2 = 4,4
np0 = np_rng.normal(size=(N0,N1)) * (np_rng.uniform(size=(N0,N1))>0.8)
torch0 = torch.tensor(np0.toarray())
torch0_csr = torch0.to_sparse_csr() #beta version
torch0_coo = torch0.to_sparse()
torch0_coo.shape #(4,4)
torch0_coo.dtype #float64
index = torch0_coo.indices() #(torch,int64,(ndim,nnz))
value = torch0_coo.values() #(torch,dtype,nnz)
# .to_sparse_bsc()
# .to_sparse_bsr()
# .to_sparse_csc()
# .to_sparse_csr()


# 3 dimension (coo default)
N0,N1,N2 = 2,3,4
np0 = np_rng.normal(size=(N0,N1,N2)) * (np_rng.uniform(size=(N0,N1,N2))>0.8)
# np0 = hf_randc(N0,N1,N2) * (np_rng.uniform(size=(N0,N1,N2))>0.8) #support complex
torch0_coo = torch.tensor(np0).to_sparse()
torch0_csr = torch.tensor(np0).to_sparse_csr()

# sparse dimensions must be the frist xxx dim
N0,N1,N2 = 4,4,4
np0 = np_rng.normal(size=(N0,N1,N2)) * (np_rng.uniform(size=(N0,1,1))>0.5)
torch0_coo = torch.tensor(np0).to_sparse(sparse_dim=1)

np0 = np_rng.normal(size=(N0,N1,N2)) * (np_rng.uniform(size=(N0,N1,1))>0.5)
torch0_coo = torch.tensor(np0).to_sparse(sparse_dim=2)


N0,N1,N2 = 3,4,5
np0 = np_rng.normal(size=(N0,N1,N2)) * (np_rng.uniform(size=(N0,N1,N2))>0.8)
index = np.stack(np.nonzero(np0)) #(np,int64,(3,nnz))
value = np0[index[0], index[1], index[2]] #(np,float64,nnz)
torch0_coo = torch.sparse_coo_tensor(index, value, (N0,N1,N2))


# 3 dimensional operator matmul fail
# N0,N1,N2 = 3,4,5
# np0 = np_rng.normal(size=(N0,N1,N2)) * (np_rng.uniform(size=(N0,N1,N2))>0.8)
# torch0_coo = torch.tensor(np0).to_sparse()
# torch1 = torch.tensor(np_rng.normal(size=(N2,23)))
# torch0_coo @ torch1 #fail


def test_scipy_sparse_csr_to_torch():
    x0 = scipy.sparse.random(3, 5, density=0.5, format='csr', dtype=np.float64)
    tmp0 = torch.tensor(x0.indptr, dtype=torch.int64)
    tmp1 = torch.tensor(x0.indices, dtype=torch.int64)
    tmp2 = torch.tensor(x0.data, dtype=torch.float64)
    torch0 = torch.sparse_csr_tensor(tmp0, tmp1, tmp2, dtype=torch.float64)
    assert np.abs(torch0.to_dense().numpy() - x0.toarray()).max() < 1e-10
