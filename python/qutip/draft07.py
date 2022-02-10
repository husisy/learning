'''
GRAPE calculation of control fields for iSWAP implementation
ref: https://nbviewer.jupyter.org/github/qutip/qutip-notebooks/blob/master/examples/control-grape-iswap.ipynb
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

np.set_printoptions(precision=3, linewidth=150)
plt.ion()

import qutip
import qutip.control
from qutip import iswap, sigmax, sigmay, sigmaz, tensor, qeye, propagator
from qutip.control.grape import plot_grape_control_fields, _overlap, grape_unitary_adaptive, cy_grape_unitary
from qutip.ui.progressbar import TextProgressBar

T = 1
times = np.linspace(0, T, 100)

U = iswap()
R = 50
H_ops = [
    tensor(sigmax(), sigmax()),
    tensor(sigmay(), sigmay()),
    tensor(sigmaz(), sigmaz())
]
H_labels = ['$u_{xx}$', '$u_{yy}$', '$u_{zz}$']
H0 = 0 * np.pi * (tensor(sigmaz(), qeye(2)) + tensor(qeye(2), sigmaz()))


# GRAPE
u0 = np.array([np.random.rand(len(times)) * (2 * np.pi / T) * 0.01 for _ in range(len(H_ops))])
u0 = [np.convolve(np.ones(10)/10, u0[idx, :], mode='same') for idx in range(len(H_ops))]
result = cy_grape_unitary(U, H0, H_ops, R, times, u_start=u0, eps=2*np.pi/T, progress_bar=TextProgressBar())

result.U_f.tidyup(1e-2) #iswap()
_overlap(U, result.U_f)

U_f_numerical = propagator(result.H_t, times, [], args={})[-1]
_overlap(U, U_f_numerical)
