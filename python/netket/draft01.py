import numpy as np
import scipy.sparse.linalg

import jax
import jax.numpy as jnp
import flax

import netket

jax_rng = jax.random.PRNGKey(0)


## variational monte carlo optimisation driver
graph = netket.graph.Hypercube(length=20, n_dim=1, pbc=True) # periodic boundary condition (pbc)

hi_space = netket.hilbert.Spin(s=1/2, N=graph.n_nodes) #spin, Fock, qubit
ha = netket.operator.Ising(hilbert=hi_space, graph=graph, h=1.0)

ma = netket.models.RBM(alpha=1, param_dtype=float) #Neural Quantum State, restricted boltzman Machine (RBM)
sa = netket.sampler.MetropolisLocal(hi_space, n_chains=16) #16 markov chains
op = netket.optimizer.Sgd(learning_rate=0.01)
gs = netket.VMC(ha, op, sa, ma, n_samples=1000, n_discard_per_chain=100)
gs.run(n_iter=300, out=None)
