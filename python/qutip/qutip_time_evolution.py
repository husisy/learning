import numpy as np

from qutip import sesolve

from utils import generate_random_operator, generate_random_ket

hfe = lambda x,y,eps=1e-3: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
hfH = lambda x: np.conjugate(x.T)


def qutip_sesolve(N0=3):
    hamiltonian = generate_random_operator(N0, is_hermite=True)
    state_initial = generate_random_ket(N0)
    time_list = np.linspace(0, 10, 20)
    operator = generate_random_operator(N0)

    ret_ = sesolve(hamiltonian, state_initial, time_list, [operator]).expect[0]

    [EVL,EVC] = np.linalg.eigh(hamiltonian.full())
    ret0 = []
    for time_i in time_list:
        tmp0 = (EVC * np.exp(-1j*EVL*time_i)) @ hfH(EVC)
        tmp1 = tmp0 @ state_initial.full()
        tmp2 = hfH(tmp1) @ operator.full() @ tmp1
        ret0.append(tmp2[0,0])
    ret0 = np.array(ret0)
    print('qutip_sesolve:: qutip vs analytic: ', hfe(ret_, ret0))


if __name__ == "__main__":
    qutip_sesolve()
