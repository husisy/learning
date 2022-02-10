'''ref: http://nbviewer.ipython.org/urls/raw.github.com/jrjohansson/qutip-lectures/master/Lecture-0-Introduction-to-QuTiP.ipynb'''
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, linewidth=150)
plt.ion()

import qutip
from qutip import qeye, sigmax, sigmay, sigmaz, create, destroy, displace, squeeze #operator
from qutip import fock, coherent, fock_dm, coherent_dm, thermal_dm, ket2dm #state
from qutip import tracedist, fidelity, hilbert_dist, bures_dist, bures_angle #metrics
from qutip import operator_to_vector, vector_to_operator, spre, spost, to_super, liouvillian, to_choi #superoperator
from qutip import Qobj, tensor, expect, commutator
from qutip import mesolve #master equation solve


## Qobj
tmp0 = Qobj([[1], [0]])
_ = Qobj(np.random.rand(2,1))
tmp0.dims
tmp0.shape
tmp0.data
tmp0.isherm
tmp0.type #ket bra oper super operator-ket operator-bra
# tmp0.superrep
tmp0.full()


## state
tmp0 = fock(4, 2) #qutip.basis(4, 2)
tmp0.dag()
tmp0.tr()
tmp0.eigenenergies()
tmp1 = fock_dm(4, 2)

tmp0 = coherent(5, 1.0)
tmp1 = coherent_dm(5, 1.0)

tmp0 = thermal_dm(5, 1)



## operator
tmp0 = sigmaz() + 0.1*sigmay()
tmp0.dag()
tmp0.tr()
tmp0.eigenenergies()

sigmax() #Qobj([[0,1], [1,0]])
sigmay() #Qobj([[0,-1j], [1j,0]])
sigmaz() #Qobj([[1,0], [0,-1]])

tmp0 = create(5)
tmp1 = destroy(5)

#truncated Hilbert space, the highest fork state should NOT be involved in the dynamics
a = destroy(5)
x = (a + a.dag()) / np.sqrt(2)
p = -1j * (a - a.dag()) / np.sqrt(2)
commutator(a, a.dag())
commutator(x, p)

tmp0 = tensor(sigmaz(), qeye(2))
tmp1 = tensor(sigmax(), sigmax())

# TODO displace + squeeze -> coherent
# see http://qutip.org/docs/latest/guide/guide-states.html#state-vectors-kets-or-bras
qutip.displace
qutip.squeeze
qutip.coherent



# Jaynes-Cumming Hamiltonian
# http://qutip.org/docs/latest/guide/guide-tensor.html#a-two-level-system-coupled-to-a-cavity-the-jaynes-cummings-model
cavity_frequency = 1
qubit_freqency = 1
coupling_strength = 0.1
tmp0 = tensor(destroy(5), qeye(2))
tmp1 = tensor(qeye(5), destroy(2))
tmp2 = cavity_frequency * tmp0.dag() * tmp0
tmp3 = -0.5 * qubit_freqency * tensor(qeye(5), sigmaz())
tmp4 = coupling_strength*(tmp0*tmp1.dag() + tmp0.dag()*tmp1)
hamiltonian = tmp2 + tmp3 + tmp4

# unitary dynamics
hamiltonian = sigmax()
initial_state = fock(2, 0)
time_list = np.linspace(0, 10, 100)
z0 = mesolve(hamiltonian, initial_state, time_list)
z0.states[-1]

fig,ax = plt.subplots(1, 1)
ax.plot(time_list, expect(sigmaz(), z0.states))
ax.set_xlabel('$t$')
ax.set_ylabel(r'$\left<\sigma_z\right>$')
plt.close(fig)

z0 = mesolve(hamiltonian, initial_state, time_list, e_ops=[sigmax(), sigmay(), sigmaz()])
fig,ax = plt.subplots(1, 1)
ax.plot(time_list, z0.expect[0], label=r'$\left<\sigma_x\right>$')
ax.plot(time_list, z0.expect[1], label=r'$\left<\sigma_y\right>$')
ax.plot(time_list, z0.expect[2], label=r'$\left<\sigma_z\right>$')
ax.set_xlabel('$t$')
ax.legend()

# dissipative dynamics
angular_frequency = 1
relaxation_rate = 0.1
tmp0 = destroy(10)
hamiltonian = angular_frequency * tmp0.dag() * tmp0
initial_state = fock_dm(10, 5) #fock state with 5 photon
collapse_operator = [np.sqrt(relaxation_rate)*tmp0]
time_list = np.linspace(0, 50, 100)
z0 = mesolve(hamiltonian, initial_state, time_list, c_ops=collapse_operator, e_ops=[tmp0.dag()*tmp0])

fig,ax = plt.subplots(1, 1)
ax.plot(time_list, z0.expect[0])
ax.set_xlabel('$t$')
ax.set_ylabel('photon number')


## superoperators and vectorized operators, see qutip_super_operator.py
tmp0 = to_super(sigmax())
tmp0.iscp
tmp0.istp
tmp0.istpcp

liouvillian(10*sigmaz())
L = liouvillian(10*sigmaz(), [destroy(2)])
S = (12*L).expm()

# TODO
# qutip.tensor
# qutip.super_tensor
# qutip.composite
# qutip.tensor_contract

