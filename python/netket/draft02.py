import os
import json
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import flax
# import flax.linen as nn

import netket

jax_rng = jax.random.PRNGKey(0)
hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file('')):
    os.makedirs(hf_file(''))

# https://netket.readthedocs.io/en/latest/tutorials/gs-heisenberg.html
# ## Heisenberg model ground state

L = 22
graph = netket.graph.Hypercube(length=L, n_dim=1, pbc=True)
# impose to have a fixed total magnetization of zero
hi_space = netket.hilbert.Spin(s=0.5, total_sz=0, N=graph.n_nodes)
ha = netket.operator.Heisenberg(hilbert=hi_space, graph=graph)

# compute the ground-state energy (here we only need the lowest energy, and do not need the eigenstate)
exact_gs_energy = netket.exact.lanczos_ed(ha, compute_eigenvectors=False)[0] #about 10 seconds
# -39.14752260706246


class JastrowModel(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, x):
        x = jnp.atleast_2d(x)
        ret = jax.vmap(self.evaluate_single, in_axes=(0))(x)
        return ret

    def evaluate_single(self, x):
        N0 = x.shape[-1]
        v_bias = self.param("visible_bias", flax.linen.initializers.normal(), (N0,), complex)
        J = self.param("kernel", flax.linen.initializers.normal(), (N0,N0), complex)
        ret = x.T@J@x + jnp.dot(x, v_bias)
        return ret

model = JastrowModel()
sa = netket.sampler.MetropolisExchange(hilbert=hi_space, graph=graph)
vstate = netket.vqs.MCState(sa, model, n_samples=1008)
# netket.utils.is_probably_holomorphic(vstate._apply_fun, vstate.parameters, vstate.samples, vstate.model_state)
gs = netket.VMC(
    hamiltonian=ha,
    optimizer=netket.optimizer.Sgd(learning_rate=0.1),
    preconditioner=netket.optimizer.SR(diag_shift=0.1),
    variational_state=vstate,
    # holomorphic=True,
)
# about 14 seconds
logfile_jastrow = hf_file('Jastrow')
gs.run(300, out=logfile_jastrow)
vstate.n_parameters
# netket.jax.tree_size(vstate.parameters)


with open(logfile_jastrow+'.log', 'r') as fid:
    tmp0 = json.load(fid)
    iter_Jastrow = tmp0['Energy']['iters']
    E_Jastrow = tmp0['Energy']['Mean']['real']


## restricted boltzmann machine (RBM)
model = netket.models.RBM(alpha=1) # #hidden = alpha * L
sa = netket.sampler.MetropolisExchange(hilbert=hi_space, graph=graph)
vstate = netket.vqs.MCState(sa, model, n_samples=1008)
gs = netket.VMC(
    hamiltonian=ha,
    optimizer=netket.optimizer.Sgd(learning_rate=0.05),
    preconditioner=netket.optimizer.SR(diag_shift=0.1),
    variational_state=vstate,
)
logfile_RBM = hf_file('RBM')
# about 30 seconds
gs.run(n_iter=600, out=logfile_RBM)
vstate.n_parameters #528

with open(logfile_RBM+'.log', 'r') as fid:
    tmp0 = json.load(fid)
    iter_RBM = tmp0['Energy']['iters']
    E_RBM = tmp0['Energy']['Mean']


## Symmetric RBM Spin Machine
model = netket.models.RBMSymm(symmetries=graph.translation_group(), alpha=1)
# Metropolis Exchange Sampling: exchanges two neighboring sites, thus preservers the total magnetization
sa = netket.sampler.MetropolisExchange(hi_space, graph=graph)
vstate = netket.vqs.MCState(sa, model, n_samples=1008)
gs = netket.VMC(
    hamiltonian=ha,
    optimizer=netket.optimizer.Sgd(learning_rate=0.05),
    preconditioner=netket.optimizer.SR(diag_shift=0.1),
    variational_state=vstate,
)
logfile_rbm_sym = hf_file('RBMSymmetric')
# about 15 seconds
gs.run(n_iter=300, out=logfile_rbm_sym) #fail to converge sometimes, just repeat
vstate.n_parameters #24

with open(logfile_rbm_sym+'.log', 'r') as fid:
    tmp0 = json.load(fid)
    iter_RBM_sym = tmp0['Energy']['iters']
    E_RBM_sym = tmp0['Energy']['Mean']


class NeuralNetworkStateModel(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, x):
        x = flax.linen.Dense(features=2*x.shape[-1], param_dtype=np.complex128,
                    kernel_init=flax.linen.initializers.normal(stddev=0.1), bias_init=flax.linen.initializers.normal(stddev=0.1))(x)
        x = netket.nn.activation.log_cosh(x)
        x = jax.numpy.sum(x, axis=-1)
        return x
ffnn = NeuralNetworkStateModel()
sa = netket.sampler.MetropolisExchange(hi_space, graph=graph)
vstate = netket.vqs.MCState(sa, ffnn, n_samples=1008)
gs = netket.VMC(
    hamiltonian=ha,
    optimizer=netket.optimizer.Sgd(learning_rate=0.05),
    preconditioner=netket.optimizer.SR(diag_shift=0.1),
    variational_state=vstate,
)
logfile_nn = hf_file('neural_network')
# about 1min
gs.run(n_iter=300, out=logfile_nn)
vstate.n_parameters #1012

with open(logfile_nn+'.log', 'r') as fid:
    tmp0 = json.load(fid)
    iter_nn = tmp0['Energy']['iters']
    E_nn = tmp0['Energy']['Mean']['real']


fig, ax = plt.subplots()
ax.plot(iter_Jastrow, E_Jastrow, label='Jastrow')
ax.plot(iter_RBM, E_RBM, label='RBM')
ax.plot(iter_RBM_sym, E_RBM_sym, label='RBMSymmetric')
ax.plot(iter_nn, E_nn, label='NN')
ax.axhline(exact_gs_energy, label='Exact')
ax.set_ylabel('Energy')
ax.set_xlabel('Iteration')
ax.set_xlim(0, iter_Jastrow[-1])
ax.set_ylim(exact_gs_energy-0.1, exact_gs_energy+0.4)
ax.legend()
fig.savefig('tbd00.png')
