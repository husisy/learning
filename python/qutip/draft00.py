# http://nbviewer.ipython.org/urls/raw.github.com/jrjohansson/qutip-lectures/master/Lecture-0-Introduction-to-QuTiP.ipynb
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import qutip
# from qutip import qeye, sigmax, sigmay, sigmaz, create, destroy, displace, squeeze #operator
# from qutip import fock, coherent, fock_dm, coherent_dm, thermal_dm, ket2dm #state
# from qutip import tracedist, fidelity, hilbert_dist, bures_dist, bures_angle #metrics
# from qutip import operator_to_vector, vector_to_operator, spre, spost, to_super, liouvillian, to_choi #superoperator
# from qutip import Qobj, tensor, expect, commutator
# from qutip import mesolve #master equation solve
# from qutip import tensor, qeye, destroy, expect, ket2dm, ptrace, wigner


## Qobj
qutip.Qobj(np.random.rand(2,1)) #not normalized
z0 = qutip.Qobj([[1], [0]])
z0.dims
z0.shape
z0.data
z0.isherm
z0.type #ket bra oper super operator-ket operator-bra
# z0.superrep
z0.full()


## state
qutip.basis(4, 2) #(np,complex128,csr,(4,1)) (0,0,1,0)
z0 = qutip.fock(4, 2) #(0,0,1,0)
z0.dag() #(np,complex128,csr,(1,4))
z0.tr()
z0.eigenenergies()
qutip.fock_dm(4, 2)
qutip.coherent(5, 1.0)
qutip.coherent_dm(5, 1.0)
qutip.thermal_dm(5, 1)


## operator
z0 = qutip.sigmaz() + 0.1*qutip.sigmay()
z0.dag()
z0.tr()
z0.eigenenergies()

qutip.sigmax() #Qobj([[0,1], [1,0]])
qutip.sigmay() #Qobj([[0,-1j], [1j,0]])
qutip.sigmaz() #Qobj([[1,0], [0,-1]])
qutip.create(5)
qutip.destroy(5)

#truncated Hilbert space, the highest fork state should NOT be involved in the dynamics
a = qutip.destroy(5)
x = (a + a.dag()) / np.sqrt(2)
p = -1j * (a - a.dag()) / np.sqrt(2)
qutip.commutator(a, a.dag())
qutip.commutator(x, p)

tmp0 = qutip.tensor(qutip.sigmaz(), qutip.qeye(2))
tmp1 = qutip.tensor(qutip.sigmax(), qutip.sigmax())

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
tmp0 = qutip.tensor(qutip.destroy(5), qutip.qeye(2))
tmp1 = qutip.tensor(qutip.qeye(5), qutip.destroy(2))
tmp2 = cavity_frequency * tmp0.dag() * tmp0
tmp3 = -0.5 * qubit_freqency * qutip.tensor(qutip.qeye(5), qutip.sigmaz())
tmp4 = coupling_strength*(tmp0*tmp1.dag() + tmp0.dag()*tmp1)
hamiltonian = tmp2 + tmp3 + tmp4

# unitary dynamics
hamiltonian = qutip.sigmax()
initial_state = qutip.fock(2, 0)
time_list = np.linspace(0, 10, 100)
z0 = qutip.mesolve(hamiltonian, initial_state, time_list)
z0.states #(list,QObj)
fig,ax = plt.subplots(1, 1)
ax.plot(time_list, qutip.expect(qutip.sigmaz(), z0.states))
ax.set_xlabel('$t$')
ax.set_ylabel(r'$\left<\sigma_z\right>$')

z0 = qutip.mesolve(hamiltonian, initial_state, time_list, e_ops=[qutip.sigmax(), qutip.sigmay(), qutip.sigmaz()])
fig,ax = plt.subplots(1, 1)
ax.plot(time_list, z0.expect[0], label=r'$\left<\sigma_x\right>$')
ax.plot(time_list, z0.expect[1], label=r'$\left<\sigma_y\right>$')
ax.plot(time_list, z0.expect[2], label=r'$\left<\sigma_z\right>$')
ax.set_xlabel('$t$')
ax.legend()

# dissipative dynamics
angular_frequency = 1
relaxation_rate = 0.1
tmp0 = qutip.destroy(10)
hamiltonian = angular_frequency * tmp0.dag() * tmp0
initial_state = qutip.fock_dm(10, 5) #fock state with 5 photon
collapse_operator = [np.sqrt(relaxation_rate)*tmp0]
time_list = np.linspace(0, 50, 100)
z0 = qutip.mesolve(hamiltonian, initial_state, time_list, c_ops=collapse_operator, e_ops=[tmp0.dag()*tmp0])
fig,ax = plt.subplots(1, 1)
ax.plot(time_list, z0.expect[0])
ax.set_xlabel('$t$')
ax.set_ylabel('photon number')


## superoperators and vectorized operators, see qutip_super_operator.py
z0 = qutip.to_super(qutip.sigmax())
z0.iscp
z0.istp
z0.istpcp

qutip.liouvillian(10*qutip.sigmaz())
L = qutip.liouvillian(10*qutip.sigmaz(), [qutip.destroy(2)])
S = (12*L).expm()

# TODO
# qutip.tensor
# qutip.super_tensor
# qutip.composite
# qutip.tensor_contract

