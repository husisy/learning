import numpy as np
import torch

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


def test_torch_mm(N0=3, N1=5, N2=7):
    np0 = np.random.rand(N0, N1)
    np1 = np.random.rand(N1, N2)
    ret_ = np0 @ np1
    ret0 = torch.mm(torch.tensor(np0), torch.tensor(np1))
    assert hfe(ret_, ret0.numpy()) < 1e-5


def test_torch_addmm(N0=3, N1=5, N2=7):
    beta = np.random.randn()
    alpha = np.random.randn()
    np0 = np.random.rand(N0, N2)
    np1 = np.random.rand(N0, N1)
    np2 = np.random.randn(N1, N2)
    ret_ = beta*np0 + alpha*(np1 @ np2)
    ret0 = torch.addmm(torch.tensor(np0), torch.tensor(np1), torch.tensor(np2), beta=beta, alpha=alpha)
    assert hfe(ret_, ret0.numpy()) < 1e-5

    ret1 = torch.tensor(np0.copy())
    ret1.addmm_(torch.tensor(np1), torch.tensor(np2), beta=beta, alpha=alpha)
    assert hfe(ret_, ret1.numpy()) < 1e-5


def test_torch_addcdiv(N0=3):
    np0 = np.random.randn(N0)
    np1 = np.random.randn(N0)
    np2 = np.random.rand(N0) + 1
    np3 = np.random.randn()
    ret_ = np0 + np3 * np1 / np2
    ret0 = torch.addcdiv(torch.tensor(np0), torch.tensor(np1), torch.tensor(np2), value=np3)
    assert hfe(ret_, ret0.numpy()) < 1e-5

    ret1 = torch.tensor(np0)
    ret1.addcdiv_(torch.tensor(np1), torch.tensor(np2), value=np3)
    assert hfe(ret_, ret1.numpy()) < 1e-5


def test_torch_addcmul(N0=3):
    np0 = np.random.randn(N0)
    np1 = np.random.randn(N0)
    np2 = np.random.randn(N0)
    np3 = np.random.randn()
    ret_ = np0 + np3*np1*np2
    ret0 = torch.addcmul(torch.tensor(np0), torch.tensor(np1), torch.tensor(np2), value=np3)
    assert hfe(ret_, ret0.numpy()) < 1e-5

    ret1 = torch.tensor(np0)
    ret1.addcmul_(torch.tensor(np1), torch.tensor(np2), value=np3)
    assert hfe(ret_, ret1.numpy()) < 1e-5


def test_torch_addbmm(N0=13, N1=3, N2=5, N3=7):
    np0 = np.random.randn(N1, N3)
    np1 = np.random.randn(N0, N1, N2)
    np2 = np.random.randn(N0, N2, N3)
    beta = np.random.randn()
    alpha = np.random.randn()
    ret_ = beta*np0 + alpha*sum(x@y for x,y in zip(np1,np2))
    ret0 = torch.addbmm(torch.tensor(np0), torch.tensor(np1), torch.tensor(np2), beta=beta, alpha=alpha)
    assert hfe(ret_, ret0.numpy()) < 1e-5

    ret1 = torch.tensor(np0)
    ret1.addbmm_(torch.tensor(np1), torch.tensor(np2), beta=beta, alpha=alpha)
    assert hfe(ret_, ret1.numpy()) < 1e-5


def test_var(N0=5, N1=7):
    np_rng = np.random.default_rng()
    np0 = np_rng.uniform(size=(N0,N1)).astype(np.float32)
    torch0 = torch.tensor(np0, dtype=torch.float32)
    ret0_ = np.var(np0, axis=1, ddof=1)
    ret1_ = np.var(np0, axis=1, ddof=0) #default

    ret0 = torch.var(torch0, dim=1, unbiased=True).numpy()
    ret1 = torch.var(torch0, dim=1, unbiased=False).numpy()
    assert hfe(ret0_, ret0) < 1e-5
    assert hfe(ret1_, ret1) < 1e-5
