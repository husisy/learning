import numpy as np
import scipy.sparse.linalg

import jax
import jax.numpy as jnp
import flax
# import flax.linen as nn

import netket

jax_rng = jax.random.PRNGKey(0)


## transverse-field Ising model ground state https://netket.readthedocs.io/en/latest/tutorials/gs-ising.html
N = 20
Gamma = -1
V = -1
hi_space = netket.hilbert.Spin(s=1/2, N=netket.operator.spin.N)
H = sum([Gamma*netket.operator.spin.sigmax(hi_space,i) for i in range(N)])
H += sum([V*netket.operator.spin.netket.operator.spinetket.operator.spin.n.sigmaz(hi_space,i)*netket.operator.spin.sigmaz(hi_space,(i+1)%N) for i in range(N)]) #tensor product
# (scipy.sparse, float64, (1048576,1048576))
EVL,EVC = scipy.sparse.linalg.eigsh(H.to_sparse(), k=2, which='SA') #about 10 seconds
## EVL: -25.49098969, -25.41240947

## mean field ansatz
class MeanFieldModel(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, x):
        lambda_ = self.param('lambda', flax.linen.initializers.normal(), (1,), float)
        # x (flax, (?,20))
        p = flax.linen.log_sigmoid(lambda_*x)
        ret = 0.5*jnp.sum(p, axis=-1)
        return ret

mf_model = MeanFieldModel()
# Create the local sampler on the hilbert space
sampler = netket.sampler.MetropolisLocal(hi_space)
# construct the variational state, `n_samples` specifies how many samples should be used to compute expectation values
vstate = netket.vqs.MCState(sampler, mf_model, n_samples=512)
vstate.parameters['lambda'] #(jax,float64)
E = vstate.expect(H) #netket.state.mc_stats.Stats
E.mean #-20.12609348
E.variance #19.80920569
E.error_of_mean #0.19669737
vstate.expect_and_grad(H)

# optimize (variational Monte carlo)
vstate.init_parameters()
optimizer = netket.optimizer.Sgd(learning_rate=0.05)
gs = netket.driver.VMC(H, optimizer, variational_state=vstate)
gs.run(n_iter=300)
E_optim = vstate.expect(H)
vstate.parameters["lambda"] #-2.63098545


class JastrowModel(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, x):
        j1 = self.param("j1", flax.linen.initializers.normal(), (1,), float)
        j2 = self.param("j2", flax.linen.initializers.normal(), (1,), float)
        tmp0 = x*jnp.roll(x,-1,axis=-1)
        tmp1 = x*jnp.roll(x,-2,axis=-1)
        ret = jnp.sum(j1*tmp0+j2*tmp1,axis=-1)
        return ret

model = JastrowModel()
vstate = netket.vqs.MCState(sampler, model, n_samples=1008)
optimizer = netket.optimizer.Sgd(learning_rate=0.05)
# natural gradient preconditioner, Stochastic Reconfiguration (SR)
gs = netket.driver.VMC(H, optimizer, variational_state=vstate, preconditioner=netket.optimizer.SR(diag_shift=0.1))
logger = netket.logging.RuntimeLog()
gs.run(n_iter=300, out=logger)
E_optim = vstate.expect(H)
logger.data['Energy'].iters
logger.data['Energy'].Sigma
logger.data['Energy']['Mean']


class NeuralNetworkStateModel(flax.linen.Module):
    alpha : int = 1
    @flax.linen.compact
    def __call__(self, x):
        # netket.flax.linen.Dense for complex hamiltonian
        dense = flax.linen.Dense(features=self.alpha * x.shape[-1])
        x = dense(x)
        x = flax.linen.relu(x)
        x = jnp.sum(x, axis=-1)
        return x

model = NeuralNetworkStateModel(alpha=1)
vstate = netket.vqs.MCState(sampler, model, n_samples=1008)
optimizer = netket.optimizer.Sgd(learning_rate=0.1)
gs = netket.driver.VMC(H, optimizer, variational_state=vstate,preconditioner=netket.optimizer.SR(diag_shift=0.1))
logger = netket.logging.RuntimeLog()
gs.run(n_iter=300,out=logger)
E_optim = vstate.expect(H)


## translation symmetry
N = 20
Gamma = -1
V = -1
graph = netket.graph.Chain(length=N, pbc=True)
graph.translation_group() #list of length N
H = sum([Gamma*netket.operator.spin.sigmax(hi_space,i) for i in range(N)])
H += sum([V*netket.operator.spin.sigmaz(hi_space,i)*netket.operator.spin.sigmaz(hi_space,j) for (i,j) in graph.edges()])


class SymmModel(flax.linen.Module):
    alpha: int #number of features per symmetry
    @flax.linen.compact
    def __call__(self, x):
        # add an extra dimension with size 1, because DenseSymm requires rank-3 tensors as inputs.  (batches, 1, Nsites)
        x = x.reshape(-1, 1, x.shape[-1])
        tmp0 = netket.nn.DenseSymm(symmetries=graph.translation_group(), features=self.alpha,
                           kernel_init=flax.linen.initializers.normal(stddev=0.01))
        x = tmp0(x)
        x = flax.linen.relu(x)
        x = jnp.sum(x,axis=(-1,-2))
        return x

sampler = netket.sampler.MetropolisLocal(hi_space)
model = SymmModel(alpha=4)
vstate = netket.vqs.MCState(sampler, model, n_samples=1008)
vstate.n_parameters #84

optimizer = netket.optimizer.Sgd(learning_rate=0.1)
gs = netket.driver.VMC(H, optimizer, variational_state=vstate,preconditioner=netket.optimizer.SR(diag_shift=0.1))
logger = netket.logging.RuntimeLog()
gs.run(n_iter=600, out=logger)
E_optim = vstate.expect(H)

# measuring other property
corr = sum([netket.operator.spin.sigmax(hi_space,i)*netket.operator.spin.sigmax(hi_space,j) for (i,j) in graph.edges()])
# correlators do not have the zero-variance property as the Hamiltonian, so require a larger number of samples to be estimated accurately
vstate.n_samples = 400000
op_expectation = vstate.expect(corr) # 10.873 ± 0.017
op_expectation_exact = EVC[:,0] @ (corr @ EVC[:,0]) #10.852248713127791
