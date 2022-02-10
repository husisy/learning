import numpy as np

import qutip
from qutip import operator_to_vector, vector_to_operator
from qutip import spre, spost, sprepost
from qutip import to_super, to_choi, to_chi, to_kraus, to_stinespring

from utils import generate_random_ket, generate_random_density_matrix, generate_random_operator

hf_randn_c = lambda *size: np.random.randn(*size) + np.random.randn(*size)*1j
hfH = lambda x: np.conjugate(x.T)
hfe = lambda x,y,eps=1e-3: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

def qutip_operator_and_vector(N0=5):
    operator0 = generate_random_operator(N0)
    ret_ = operator0.full().T.reshape(-1)
    ret0 = operator_to_vector(operator0).full()[:,0]
    print('qutip_operator_to_vector:: np vs qutip: ', hfe(ret_, ret0))

    tmp0 = operator_to_vector(operator0)
    ret_ = tmp0.full().reshape(*[x[0] for x in tmp0.dims[0]]).T
    ret0 = vector_to_operator(tmp0).full()
    print('qutip_vector_to_operator:: np vs qutip: ', hfe(ret_, ret0))


def qutip_spre_spost(N0=3):
    operator0 = generate_random_operator(N0)
    ret_ = np.kron(np.eye(N0), operator0.full())
    ret0 = spre(operator0).full()
    print('qutip_spre:: np vs qutip: ', hfe(ret_, ret0))

    ret_ = np.kron(operator0.full().T, np.eye(N0))
    ret0 = spost(operator0).full()
    print('qutip_spost:: np vs qutip: ', hfe(ret_, ret0))

    ret_ = np.kron(np.conjugate(operator0.full()), operator0.full())
    ret0 = to_super(operator0).full() #spre(operator0)*spost(operator0.dag())
    print('qutip_to_super:: np vs qutip: ', hfe(ret_, ret0))

    operator0 = generate_random_operator(N0)
    operator1 = generate_random_operator(N0)
    ret_ = np.kron(operator1.full().T, operator0.full())
    ret0 = sprepost(operator0, operator1).full()
    print('qutip_sprepost:: np vs qutip: ', hfe(ret_, ret0))


def qutip_choi(N0=3):
    operator0 = to_super(generate_random_operator(N0))
    ret_ = operator0.full().reshape(N0,N0,N0,N0).transpose(3,1,2,0).reshape(N0**2,N0**2)
    ret0 = to_choi(operator0).full()
    print('qutip_to_choi:: np vs qutip: ', hfe(ret_, ret0))


# TODO to_kraus
# TODO to_stinespring
# TODO to_chi
to_chi

if __name__ == "__main__":
    qutip_operator_and_vector()
    print()
    qutip_spre_spost()
    print()
    qutip_choi()
