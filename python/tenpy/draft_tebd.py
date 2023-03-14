import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib


import tenpy
import tenpy.linalg.np_conserved as npc
# from tenpy.algorithms import tebd
# from tenpy.networks.mps import MPS
# from tenpy.models.tf_ising import TFIChain

# tenpy.tools.misc.setup_logging(to_stdout="INFO")

L = 30
model_params = {
    'J': 1. , 'g': 1.,  # critical
    'L': L,
    'bc_MPS': 'finite',
}
M = tenpy.models.tf_ising.TFIChain(model_params)

psi = tenpy.networks.mps.MPS.from_lat_product_state(M.lat, [['up']])

tebd_params = {
    'N_steps': 1,
    'dt': 0.1,
    'order': 4,
    'trunc_params': {'chi_max': 100, 'svd_min': 1.e-12}
}
eng = tenpy.algorithms.tebd.TEBDEngine(psi, M, tebd_params)
# Time Evolution Block Decimation

def measurement(eng, data):
    keys = ['t', 'entropy', 'Sx', 'Sz', 'corr_XX', 'corr_ZZ', 'trunc_err']
    if data is None:
        data = dict([(k, []) for k in keys])
    data['t'].append(eng.evolved_time)
    data['entropy'].append(eng.psi.entanglement_entropy())
    data['Sx'].append(eng.psi.expectation_value('Sigmax'))
    data['Sz'].append(eng.psi.expectation_value('Sigmaz'))
    data['corr_XX'].append(eng.psi.correlation_function('Sigmax', 'Sigmax'))
    data['trunc_err'].append(eng.trunc_err.eps)
    return data

data = measurement(eng, None)

# about 2 minute in laptop
while eng.evolved_time < 5.:
    eng.run()
    measurement(eng, data)


fig,ax = plt.subplots()
ax.plot(data['t'], np.array(data['entropy'])[:, L//2])
ax.set_xlabel('time $t$')
ax.set_ylabel('entropy $S$')


fig,ax = plt.subplots()
ax.plot(data['t'], np.sum(data['Sx'], axis=1), label="X")
ax.plot(data['t'], np.sum(data['Sz'], axis=1), label="Z")
ax.set_xlabel('time $t$')
ax.set_ylabel('magnetization')
ax.legend()


corrs = np.array(data['corr_XX'])
tmax = data['t'][-1]
x = np.arange(L)
fig,ax = plt.subplots()
for i, t in list(enumerate(data['t'])):
    if i == 0 or i == len(data['t']) - 1:
        label = '{t:.2f}'.format(t=t)
    else:
        label = None
    ax.plot(x, corrs[i, L//2, :], color=matplotlib.cm.viridis(t/tmax), label=label)
ax.set_xlabel(r'time $t$')
ax.set_ylabel(r'correlations $\langle X_i X_{j:d}\rangle$'.format(j=L//2))
ax.set_yscale('log')
ax.set_ylim(1.e-4, 1.)
ax.legend()


fig,ax = plt.subplots()
ax.plot(data['t'], data['trunc_err'])
ax.set_yscale('log')
#plt.ylim(1.e-15, 1.)
ax.set_xlabel('time $t$')
ax.set_ylabel('truncation error')
