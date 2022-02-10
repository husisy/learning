'''
Calculation of control fields for state-to-state transfer of a 2 qubit system using CRAB algorithm
CRAB
ref: https://nbviewer.jupyter.org/github/qutip/qutip-notebooks/blob/master/examples/control-pulseoptim-CRAB-2qubitInerac.ipynb
'''
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

np.set_printoptions(precision=3, linewidth=150)
plt.ion()

import qutip
from qutip import Qobj, qeye, sigmax, sigmay, sigmaz, tensor, fock
from qutip.control.pulseoptim import opt_pulse_crab_unitary
import qutip.logging_utils

alpha = np.random.rand(2)
beta = np.random.rand(2)

Sx = sigmax()
Sz = sigmaz()

H_d = alpha[0]*tensor(Sx,qeye(2)) + alpha[1]*tensor(qeye(2),Sx) + beta[0]*tensor(Sz,qeye(2)) + beta[1]*tensor(qeye(2),Sz)
H_c = [tensor(Sz,Sz)]
n_ctrls = len(H_c) #number of ctrls
psi_0 = fock(4, 0)
psi_targ = fock(4, 3)

n_ts = 100 # Number of time slots
evo_time = 18 # Time allowed for the evolution
p_type = 'DEF' # pulse type alternatives: RND|ZERO|LIN|SINE|SQUARE|SAW|TRIANGLE|

# Nelder-Mead algorithm
result = opt_pulse_crab_unitary(H_d, H_c, psi_0, psi_targ, n_ts, evo_time,
            fid_err_targ=1e-3, max_iter=500, max_wall_time=120,
            init_coeff_scaling=5.0, num_coeffs=5, method_params={'xtol':1e-3},
            guess_pulse_type=None, guess_pulse_action='modulate', log_level=qutip.logging_utils.INFO, gen_stats=True)


result.stats.report()
print("Final evolution\n{}\n".format(result.evo_full_final))
print("Final fidelity error {}".format(result.fid_err))
print("Final gradient normal {}".format(result.grad_norm_final))
print("Terminated due to {}".format(result.termination_reason))
print("Number of iterations {}".format(result.num_iter))
print("Completed in {} HH:MM:SS.US".format(datetime.timedelta(seconds=result.wall_time)))

fig,(ax0,ax1) = plt.subplots(2, 1)
ax0.set_title("Initial Control amps")
ax0.set_ylabel("Control amplitude")
ax0.step(result.time, np.hstack((result.initial_amps[:, 0], result.initial_amps[-1, 0])), where='post')

ax1.set_title("Optimised Control Amplitudes")
ax1.set_xlabel("Time")
ax1.set_ylabel("Control amplitude")
ax1.step(result.time, np.hstack((result.final_amps[:, 0], result.final_amps[-1, 0])), where='post')
plt.tight_layout()
plt.show()
