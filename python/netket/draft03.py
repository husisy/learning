import os
import json
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import flax
# import flax.linen as nn

import netket

# https://netket.readthedocs.io/en/latest/tutorials/gs-j1j2.html
# J1-J2 model ground state

jax_rng = jax.random.PRNGKey(0)
hf_file = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_file('')):
    os.makedirs(hf_file(''))


#Couplings J1 and J2
J = [1, 0.2]
L = 14

tmp0 = [(x,(x+y)%L,y) for x in range(L) for y in range(2)]
graph = netket.graph.Graph(edges=tmp0)
hi_space = netket.hilbert.Spin(s=0.5, total_sz=0.0, N=graph.n_nodes)

# Custom Hamiltonian operator
#Sigma^z*Sigma^z interactions
sigmaz = [[1, 0], [0, -1]]
mszsz = np.kron(sigmaz, sigmaz)
exchange = np.array([[0, 0, 0, 0], [0, 0, 2, 0], [0, 2, 0, 0], [0, 0, 0, 0]])
tmp0 = [
    (J[0] * mszsz, 1),
    (J[1] * mszsz, 2),
    (-J[0] * exchange, 1),
    (J[1] * exchange, 2),
]
tmp1 = [x[0].tolist() for x in tmp0]
tmp2 = [x[1] for x in tmp0]
ha = netket.operator.GraphOperator(hi_space, graph=graph, bond_ops=tmp1, bond_ops_colors=tmp2)
op = ha


# We need to specify the local operators as a matrix acting on a local Hilbert space
tmp0 = [(netket.operator.spin.sigmaz(hi_space, i)*netket.operator.spin.sigmaz(hi_space, j))*((-1)**(i-j))/L for i in range(L) for j in range(L)]
structure_factor = netket.operator.LocalOperator(hi_space, dtype=complex) + sum(tmp0)


E_gs, ket_gs = netket.exact.lanczos_ed(ha, compute_eigenvectors=True) #-25.05419813
structure_factor_gs = (ket_gs.T.conj() @ structure_factor.to_linear_operator() @ ket_gs).real[0,0] #4.123629948596189


class NeuralNetworkStateModel(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, x):
        x = flax.linen.Dense(features=2*x.shape[-1],
                     use_bias=True,
                     param_dtype=np.complex128,
                     kernel_init=flax.linen.initializers.normal(stddev=0.01),
                     bias_init=flax.linen.initializers.normal(stddev=0.01)
                    )(x)
        x = netket.nn.log_cosh(x)
        x = jnp.sum(x, axis=-1)
        return x
model = NeuralNetworkStateModel()

# exchange Sampler: preserves the global magnetization (as this is a conserved quantity in the model)
sa = netket.sampler.MetropolisExchange(hilbert=hi_space, graph=graph, d_max = 2)
vstate = netket.vqs.MCState(sa, model, n_samples=1008)
gs = netket.VMC(
    hamiltonian=ha,
    optimizer=netket.optimizer.Sgd(learning_rate=0.01),
    preconditioner=netket.optimizer.SR(diag_shift=0.01),
    variational_state=vstate,
)
logfile = hf_file('J1J2')
# about 80 seconds
gs.run(n_iter=600, obs={'Structure Factor': structure_factor}, out=logfile)


with open(logfile+'.log', 'r') as fid:
    tmp0 = json.load(fid)
    iter_nn = tmp0['Energy']['iters']
    E_nn = tmp0['Energy']['Mean']['real']
    sf_nn = tmp0['Structure Factor']['Mean']['real']

fig, ax1 = plt.subplots()
ax1.plot(iter_nn, E_nn, label='Energy')
ax1.set_xlabel('Iteration')
ax2 = ax1.twinx()
ax2.plot(iter_nn, sf_nn, color='r', label='Structure Factor')
ax1.legend(loc=2)
ax2.legend(loc=1)
fig.tight_layout()
fig.savefig('tbd00.png')
