import numpy as np
import scipy.linalg

import qutip
from qutip import Qobj, ket2dm
from qutip import fidelity, tracedist, average_gate_fidelity

from utils import generate_random_ket, generate_random_density_matrix, generate_random_operator

hf_randn_c = lambda *size: np.random.randn(*size) + np.random.randn(*size)*1j
hfH = lambda x: np.conjugate(x.T)
hfe = lambda x,y,eps=1e-3: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


def qutip_fidelity(N0=5):
    ket0 = generate_random_ket(N0)
    ket1 = generate_random_ket(N0)
    ret_ = np.abs((ket0.dag()*ket1).full()[0,0])
    ret0 = fidelity(ket0, ket1)
    print('qutip_fidelity(ket):: np vs qutip: ', hfe(ret_, ret0))

    dm0 = generate_random_density_matrix(N0)
    dm1 = generate_random_density_matrix(N0)

    tmp0 = scipy.linalg.sqrtm(dm0.full())
    tmp1 = np.linalg.eigvals(tmp0 @ dm1.full() @ tmp0).real
    ret_ = np.sqrt(tmp1[tmp1>0]).sum()
    ret0 = fidelity(dm0, dm1)
    print('qutip_fidelity(density_matrix):: np vs qutip: ', hfe(ret_, ret0))


def qutip_tracedict(N0=5):
    ket0 = generate_random_ket(N0)
    ket1 = generate_random_ket(N0)
    fidelity = np.abs((ket0.dag()*ket1).full()[0,0])
    ret_ = np.sqrt(1-fidelity**2)
    ret0 = tracedist(ket0, ket1)
    print('qutip_tracedict(ket):: np vs qutip: ', hfe(ret_, ret0))
    # TODO density matrix


# TODO average_gate_fidelity


if __name__ == "__main__":
    qutip_fidelity()
    print()
    qutip_tracedict()
