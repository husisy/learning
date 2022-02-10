'''
visualization
ref: https://nbviewer.jupyter.org/github/qutip/qutip-notebooks/blob/master/examples/visualization-exposition.ipynb
'''
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, linewidth=150)
plt.ion()

import qutip
from qutip import tensor, sigmax, sigmay, sigmaz, qeye, mesolve, sesolve, expect, ket2dm, spin_q_function
from qutip import fock, rand_dm, coherent, spin_state, spin_coherent, destroy, create
from qutip.visualization import (hinton, sphereplot, matrix_histogram, plot_energy_levels, plot_fock_distribution,
        plot_wigner, plot_wigner_fock_distribution, plot_expectation_values, plot_spin_distribution_2d, plot_spin_distribution_3d)
from qutip import orbital, Bloch


hinton(rand_dm(5))
plt.close(plt.gcf())


theta = np.linspace(0, np.pi, 90)
phi = np.linspace(0, 2*np.pi, 60)
fig = plt.figure(figsize=(12,5))
ax0 = fig.add_subplot(1, 3, 1, projection='3d')
sphereplot(theta, phi, orbital(theta, phi, fock(3,0)), fig, ax0)
ax1 = fig.add_subplot(1, 3, 2, projection='3d')
sphereplot(theta, phi, orbital(theta, phi, fock(3,1)), fig, ax1)
ax2 = fig.add_subplot(1, 3, 3, projection='3d')
sphereplot(theta, phi, orbital(theta, phi, fock(3,2)), fig, ax2)
plt.close(plt.gcf())


matrix_histogram(rand_dm(5).full().real)
plt.close(plt.gcf())


hamiltonian = tensor(sigmaz(), qeye(2)) + tensor(qeye(2), sigmaz())
ham_interaction = 0.1 * tensor(sigmax(), sigmax())
plot_energy_levels([hamiltonian, ham_interaction], figsize=(8,4))
plt.close(plt.gcf())


plot_fock_distribution((coherent(15, 1.5) + coherent(15, -1.5)).unit())
plt.close(plt.gcf())


plot_wigner((coherent(15, 1.5) + coherent(15, -1.5)).unit())
plt.close(plt.gcf())


plot_wigner_fock_distribution((coherent(15, 1.5) + coherent(15, -1.5)).unit())
plt.close(plt.gcf())


hamiltonian = sigmaz() + 0.3*sigmay()
time_list = np.linspace(0, 10, 100)
initial_state = (fock(2,0) + fock(2,1)).unit()
e_ops = [sigmax(),sigmay(),sigmaz()]
z0 = mesolve(hamiltonian, initial_state, time_list, e_ops=e_ops)
plot_energy_levels(z0)
plt.close(plt.gcf())

z1 = Bloch()
z1.add_vectors(expect(hamiltonian.unit(), e_ops))
z1.add_points(z0.expect, meth='l')
z1.make_sphere()
plt.close(plt.gcf())


j = 5
psi = spin_state(j, -j)
psi = spin_coherent(j, np.pi*np.random.rand(), 2*np.pi*np.random.rand())
rho = ket2dm(psi)
theta = np.linspace(0, np.pi, 50)
phi = np.linspace(0, 2*np.pi, 50)
Q, THETA, PHI = spin_q_function(psi, theta, phi)

plot_spin_distribution_2d(Q, THETA, PHI)
plt.close(plt.gcf())

fig, ax = plot_spin_distribution_3d(Q, THETA, PHI)
plt.close(plt.gcf())


# energy-level of a quantum system as a function of a single parameter
# energy spectrum of three coupled qubits
# https://nbviewer.jupyter.org/github/qutip/qutip-notebooks/blob/master/examples/energy-levels.ipynb
atom2_frequency = 1.8*np.pi
atom3_frequency = 2.2*np.pi
coupling_atom12 = 0.1*np.pi
coupling_atom13 = 0.1*np.pi
atom1_frequency_list = np.linspace(1.5*np.pi, 2.5*np.pi, 50)

sz1 = tensor(sigmaz(), qeye(2), qeye(2))
sx1 = tensor(sigmax(), qeye(2), qeye(2))
sz2 = tensor(qeye(2), sigmaz(), qeye(2))
sx2 = tensor(qeye(2), sigmax(), qeye(2))
sz3 = tensor(qeye(2), qeye(2), sigmaz())
sx3 = tensor(qeye(2), qeye(2), sigmax())

eigenvalue_list = []
for atom1_frequency in atom1_frequency_list:
    H = atom1_frequency*sz1 + atom2_frequency*sz2 + atom3_frequency*sz3 + coupling_atom12*sx1*sx2 + coupling_atom13*sx1*sx3
    eigenvalue_list.append(H.eigenenergies()/(2*np.pi))
eigenvalue_list = np.stack(eigenvalue_list)

fig,ax = plt.subplots(figsize=(12,6))
ax.plot(atom1_frequency_list/(2*np.pi), eigenvalue_list)


# block-sphere animation TODO


# Landau-Zener evolution TODO
# https://nbviewer.jupyter.org/github/qutip/qutip-notebooks/blob/master/examples/bloch_sphere_with_colorbar.ipynb
# delta = np.pi
# sweep_rate = 4*np.pi
# hamiltonian0 = delta/2 * sigmax()
# hamiltonian1 = sweep_rate/2 * sigmaz()
# initial_state = fock(2, 0)

# sm = destroy(2)
# sx = sigmax()
# sy = sigmay()
# sz = sigmaz()
# expect_ops = [sm.dag()*sm, sx, sy, sz]
# time_list = np.linspace(-10, 10, 1500)
# expect_values = sesolve(hamiltonian0, initial_state, time_list, expect_ops).expect


