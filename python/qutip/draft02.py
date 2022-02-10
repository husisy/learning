'''https://nbviewer.jupyter.org/github/qutip/qutip-notebooks/blob/master/examples/ultrastrong-coupling-groundstate.ipynb'''
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.set_printoptions(precision=3, linewidth=150)
plt.ion()

import qutip
from qutip import tensor, qeye, destroy, expect, ket2dm, ptrace, wigner

## parameter
cavity_frequency = 2*np.pi
atom_frequency = 2*np.pi
num_state = 20
coupling_strength_list = np.linspace(0, 2.5, 50)
use_rwa = False #Set to True to see that non-RWA is necessary in this regime

## operator
a  = tensor(destroy(num_state), qeye(2))
sm = tensor(qeye(num_state), destroy(2))
na = sm.dag() * sm
nc = a.dag() * a

## calculation
na_expectation = []
nc_expectation = []
for coupling_strength in coupling_strength_list:
    if use_rwa:
        H = cavity_frequency*nc + atom_frequency*na + 2*np.pi*coupling_strength*(a.dag()*sm + a*sm.dag())
    else:
        H = cavity_frequency*nc + atom_frequency*na + 2*np.pi*coupling_strength*(a.dag()+a) * (sm+sm.dag())
    groud_state = H.eigenstates()[1][0]
    na_expectation.append(expect(na, groud_state))
    nc_expectation.append(expect(nc, groud_state))
na_expectation = np.array(na_expectation)
nc_expectation = np.array(nc_expectation)
rhoss_final = groud_state


## plot the cavity and atom occupation numbers as a function of
fig,ax = plt.subplots(figsize=(8,6))
ax.plot(coupling_strength_list, nc_expectation, label='cavity')
ax.plot(coupling_strength_list, na_expectation, label='atom excited state')
ax.legend()
ax.set_xlabel('coupling strength')
ax.set_ylabel('Occupation probability')
ax.set_title('# photons in the groundstate')


## plot the cavity wigner function for the cavity state (final coupling strenght)
fig = plt.figure(2, figsize=(9, 6))
rho_cavity = ptrace(rhoss_final, 0)
tmp0 = np.linspace(-7.5, 7.5, 100)
X,Y = np.meshgrid(tmp0, tmp0)
W = wigner(rho_cavity, tmp0, tmp0)
ax = Axes3D(fig, azim=-107, elev=49)
surf=ax.plot_surface(X, Y, W, rstride=1, cstride=1, cmap=plt.cm.RdBu, alpha=1.0, linewidth=0.05, vmax=0.25, vmin=-0.25)
ax.set_xlim3d(-7.5, 7.5)
ax.set_ylim3d(-7.5, 7.5)
fig.colorbar(surf, shrink=0.65, aspect=20)
ax.set_title('Wigner function for the cavity groundstate\n(ultra-strong coupling to a qubit)')
