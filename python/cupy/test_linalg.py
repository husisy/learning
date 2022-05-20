import cupy as cp
import torch
import numpy as np

np_rng = np.random.default_rng()

hf_torch_to_cp = lambda x: cp.from_dlpack(torch.utils.dlpack.to_dlpack(x))
hf_cp_to_torch = lambda x: torch.utils.dlpack.from_dlpack(x.toDlpack())

def cupy_linalg_eig(cp0):
    # cupy.linalg.eig is missing https://github.com/cupy/cupy/issues/3255
    torch0 = hf_cp_to_torch(cp0)
    EVL,EVC = torch.linalg.eig(torch0)
    EVL = hf_torch_to_cp(EVL)
    EVC = hf_torch_to_cp(EVC)
    return EVL, EVC


def test_cupy_linalg_eig():
    N0 = 5
    np0 = np_rng.normal(size=(N0,N0)) + 1j*np_rng.normal(size=(N0,N0))
    cp0 = cp.array(np0, dtype=cp.complex128)

    EVL,EVC = cupy_linalg_eig(cp0)
    tmp0 = cp0 @ EVC - EVL*EVC
    assert cp.abs(tmp0).max() < 1e-7
