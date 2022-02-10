'''
GRAPE calculation of control fields for single-qubit rotation
ref: https://nbviewer.jupyter.org/github/qutip/qutip-notebooks/blob/master/examples/control-grape-single-qubit-rotation.ipynb
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

np.set_printoptions(precision=3, linewidth=150)
plt.ion()

import qutip
from qutip import rx, ry, rz, sigmax, sigmay, sigmaz, propagator
from qutip.control.grape import plot_grape_control_fields, _overlap, cy_grape_unitary, grape_unitary_adaptive
from qutip.ui.progressbar import TextProgressBar

T = 1
times = np.linspace(0, T, 50)
theta, phi = np.random.rand(2)
U = rz(phi) * rx(theta)

R = 150
H_ops = [sigmax(), sigmay(), sigmaz()]
H_labels = ['$u_{x}$', '$u_{y}$' '$u_{z}$']
H0 = 0 * np.pi * sigmaz()


## GRAPE
u0 = np.array([np.random.rand(len(times)) * 2 * np.pi * 0.005 for _ in range(len(H_ops))])
u0 = [np.convolve(np.ones(10)/10, u0[idx,:], mode='same') for idx in range(len(H_ops))]
result = cy_grape_unitary(U, H0, H_ops, R, times, u_start=u0, eps=2*np.pi/T, phase_sensitive=False, progress_bar=TextProgressBar())

# plot_grape_control_fields(times, result.u[:,:,:] / (2 * np.pi), H_labels, uniform_axes=True) #FAIL strange
_overlap(U, result.U_f).real, abs(_overlap(U, result.U_f))**2

U_f_numerical = propagator(result.H_t, times, args={})[-1]
_overlap(U, U_f_numerical)
