import numpy as np
import scipy.integrate

from qutip import sigmax, sigmay, sigmaz
from qutip import fock
from qutip import mesolve, sesolve

from utils import generate_random_operator, generate_random_ket

hfe = lambda x,y: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))
hfH = lambda x: np.conjugate(x.T)


N0 = 3
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


# TODO rewrite a sesolve, use a random operator
# TODO continue qutip documentation
# TODO test pip install local
# contact FumingXu
# contact XianhuZha
# format HODLR

# def hf1(t, y, hamiltonian=hamiltonian.full()):
#     return -1j * hamiltonian @ y
# hf2 = lambda x, operator=operator.full(): (hfH(x[:,np.newaxis]) @ operator @ x[:,np.newaxis])[0,0]
# tmp0 = hfH(state_initial.full()) @ operator.full() @ state_initial.full()
# ret1 = [hf2(state_initial.full()[:,0])]

# z0 = scipy.integrate.ode(hf1)
# z0.set_integrator('zvode', method='adams')
# z0.set_initial_value(time_list[0], state_initial.full()[:,0])
# for time_i in time_list[1:]:
#     if not z0.successful():
#         break
#     tmp0 = z0.integrate(time_i)
#     ret1.append(hf2(tmp0))
