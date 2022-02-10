'''
GRAPE calculation of control fields for CNOT implementation
ref: https://nbviewer.jupyter.org/github/qutip/qutip-notebooks/blob/master/examples/control-grape-cnot.ipynb
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

np.set_printoptions(precision=3, linewidth=150)
plt.ion()

import qutip
from qutip import cnot, sigmax, sigmay, sigmaz, tensor, qeye, propagator, Odeoptions
from qutip.control.grape import plot_grape_control_fields, _overlap, grape_unitary_adaptive, cy_grape_unitary
from qutip.ui.progressbar import TextProgressBar

T = 2 * np.pi
times = np.linspace(0, T, 50)

U = cnot()
R = 500
H_ops = [
    tensor(sigmax(), qeye(2)),
    tensor(sigmay(), qeye(2)),
    tensor(sigmaz(), qeye(2)),
    tensor(qeye(2), sigmax()),
    tensor(qeye(2), sigmay()),
    tensor(qeye(2), sigmaz()),
    tensor(sigmax(),sigmax()) + tensor(sigmay(),sigmay()) + tensor(sigmaz(),sigmaz())
]
H_labels = ['$u_{1x}$', '$u_{1y}$', '$u_{1z}$', '$u_{2x}$', '$u_{1y}$', '$u_{2z}$', '$u_{xx}$', '$u_{yy}$', '$u_{zz}$']

H0 = 0 * np.pi * (tensor(sigmax(), qeye(2)) + tensor(qeye(2), sigmax()))
c_ops = []

# This is the analytical result in the absense of single-qubit tunnelling
#g = pi/(4 * T)
#H = g * (tensor(sigmax(), sigmax()) + tensor(sigmay(), sigmay()))

## GRAPE
u0 = np.array([np.random.rand(len(times)) * 2 * np.pi * 0.05 for _ in range(len(H_ops))])
u0 = [np.convolve(np.ones(10)/10, u0[idx,:], mode='same') for idx in range(len(H_ops))]
u_limits = None #[0, 1 * 2 * pi]
alpha = None
result = cy_grape_unitary(U, H0, H_ops, R, times, u_start=u0, u_limits=u_limits,
            eps=2*np.pi*1, alpha=alpha, phase_sensitive=False, progress_bar=TextProgressBar())

plot_grape_control_fields(times, result.u / (2 * np.pi), H_labels, uniform_axes=True)

U_f_numerical = propagator(result.H_t, times, [], options=Odeoptions(nsteps=500), args={})[-1]
_overlap(result.U_f, U_f_numerical).real, abs(_overlap(result.U_f, U_f_numerical))**2
