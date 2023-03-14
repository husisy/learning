import numpy as np
import torch

np_rng = np.random.default_rng()

def test_sparse_matmul_gradient():
    N0,N1,N2 = 3,5,4
    np0 = np_rng.normal(size=(N0,N1)) * (np_rng.uniform(size=(N0,N1))>0.5)
    np1 = np_rng.normal(size=(N1,N2))
    np2 = np_rng.normal(size=(N0,N2))
    torch0_coo = torch.tensor(np0).to_sparse()
    torch1 = torch.tensor(np1, requires_grad=True)
    torch2 = torch.tensor(np2)
    loss = ((torch0_coo @ torch1)*torch2).sum()
    loss.backward()
    ret_ = np0.T @ np2
    assert np.abs(ret_ - torch1.grad.detach().numpy()).max() < 1e-10


def test_sparse_matv_gradient():
    N0,N1 = 3,5
    np0 = np_rng.normal(size=(N0,N1)) * (np_rng.uniform(size=(N0,N1))>0.5)
    np1 = np_rng.normal(size=N1)
    np2 = np_rng.normal(size=N0)
    torch0_coo = torch.tensor(np0).to_sparse()
    torch1 = torch.tensor(np1, requires_grad=True)
    torch2 = torch.tensor(np2)
    loss = ((torch0_coo @ torch1)*torch2).sum()
    loss.backward()
    ret_ = np0.T @ np2
    assert np.abs(ret_ - torch1.grad.detach().numpy()).max() < 1e-10


def test_sparse_index_reshape():
    N0,N1,N2 = 3,5,4
    np0 = np_rng.normal(size=(N0,N1,N2)) * (np_rng.uniform(size=(N0,N1,N2))>0.7)
    torch0_coo = torch.tensor(np0).to_sparse()

    index = torch0_coo.indices()
    shape = torch0_coo.shape
    tmp0 = torch.stack([index[0]*shape[1]+index[1], index[2]]) #(dim,nnz)
    torch1_coo = torch.sparse_coo_tensor(tmp0, torch0_coo.values(), (shape[0]*shape[1],shape[2]))

    ret0 = torch1_coo.to_dense().numpy()
    assert np.abs(np0.reshape(N0*N1,N2)-ret0).max() < 1e-10
