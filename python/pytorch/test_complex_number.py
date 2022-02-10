import numpy as np
import torch

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
hf_randc = lambda *size: np.random.randn(*size) + 1j*np.random.randn(*size)

def test_complex_basic():
    N0 = 5
    np0 = hf_randc(N0, N0)
    torch0 = torch.tensor(np0, dtype=torch.complex128, requires_grad=True)
    loss = torch0.real.sum() + torch0.imag.sum()
    loss.backward()
    ret0 = torch0.grad.detach().numpy().copy()
    ret_ = np.ones((N0, N0)) * (1+1j)
    assert hfe(ret_, ret0) < 1e-7

    N0 = 5
    np0 = hf_randc(N0, N0)
    np1 = hf_randc(N0, N0)
    torch0 = torch.tensor(np0, dtype=torch.complex128, requires_grad=True)
    torch1 = torch.tensor(np1, dtype=torch.complex128)
    tmp0 = torch0 * torch1
    loss = tmp0.real.sum() + tmp0.imag.sum()
    loss.backward()
    ret0 = torch0.grad.detach().numpy().copy()
    ret_ = (1+1j) * np1.conj()
    assert hfe(ret_, ret0) < 1e-7


def test_real_matrix_inverse():
    # https://math.stackexchange.com/q/1471825
    # also see draft_complex_derivative.afx
    N0 = 5
    np0 = np.eye(N0) + np.random.rand(N0,N0)
    np1 = np.random.randn(N0,N0)
    torch0 = torch.tensor(np0, dtype=torch.float64, requires_grad=True)
    loss = (torch.linalg.inv(torch0)*torch.tensor(np1, dtype=torch.float64)).sum()
    loss.backward()

    ret_ = -np.linalg.inv(np0).T @ np1 @ np.linalg.inv(np0).T
    ret0 = torch0.grad.detach().numpy().copy()
    assert hfe(ret_,ret0) < 1e-7


def test_complex_matrix_inverse():
    # also see draft_complex_derivative.afx
    N0 = 5
    np0 = np.eye(N0) + np.random.rand(N0,N0) + 1j*np.random.rand(N0,N0)
    np1 = np.random.randn(N0,N0) + 1j*np.random.rand(N0,N0)
    torch0 = torch.tensor(np0, dtype=torch.complex128, requires_grad=True)
    torch1 = torch.tensor(np1, dtype=torch.complex128)
    tmp0 = torch.linalg.inv(torch0)
    loss = (tmp0.real*torch1.real).sum() + (tmp0.imag*torch1.imag).sum()
    loss.backward()

    ret0 = torch0.grad.detach().numpy().copy()
    tmp0 = np.linalg.inv(np0).T.conj()
    ret_ = - tmp0 @ np1 @ tmp0
    assert hfe(ret_, ret0) < 1e-7
