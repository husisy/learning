import numpy as np
import scipy.linalg

import picos

np_rng = np.random.default_rng()
hf_randc = lambda *x: np_rng.uniform(-1,1,size=x) + 1j*np_rng.uniform(-1,1,size=x)


def get_fidelity_np(np0, np1):
    assert np.abs(np0-np0.T.conj()).max() < 1e-10
    assert np.abs(np1-np1.T.conj()).max() < 1e-10
    tmp0 = scipy.linalg.sqrtm(np0)
    ret = np.sqrt(np.maximum(0, np.linalg.eigvalsh(tmp0 @ np1 @ tmp0))).sum()
    return ret

def get_fidelity_picos(np0, np1):
    prob = picos.Problem()
    Z = picos.ComplexVariable("Z", np0.shape)
    prob.set_objective("max", 0.5*picos.trace(Z + Z.H))
    tmp0 = picos.Constant('P',np0)
    tmp1 = picos.Constant('Q',np1)
    prob.add_constraint(((tmp0 & Z) // (Z.H & tmp1)) >> 0)
    prob.solve(solver = "cvxopt")
    return prob.value, Z.np


def test_PSD_fidelity():
    # https://picos-api.gitlab.io/picos/complex.html
    N0 = 4
    tmp0 = hf_randc(N0, N0)
    np0 = tmp0 @ tmp0.T.conj()
    tmp0 = hf_randc(N0, N0)
    np1 = tmp0 @ tmp0.T.conj()

    ret_ = get_fidelity_np(np0, np1)
    ret0,Z = get_fidelity_picos(np0, np1)
    assert abs(ret_-ret0) < 1e-7
