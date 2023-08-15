# https://netket.readthedocs.io/en/latest/tutorials/gs-matrix-models.html
# ground state of bosonic matrix model
import os
import json
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import flax
# import flax.linen as nn

import netket
# import netket.tools.info


Lambda = 4  # cutoff of each bosonic Fock space
N = 6  # number of bosons

hi_space = netket.hilbert.Fock(n_max=Lambda-1, N=N)  # n_max -> Maximum occupation for a site (inclusive)
hi_space.shape #(4,4,4,4,4,4)
hi_space.size #6
hi_space.n_states #4096

H_free = sum([0.5 + netket.operator.boson.create(hi_space, i) * netket.operator.boson.destroy(hi_space, i) for i in range(N)])

eig_vals = netket.exact.lanczos_ed(H_free, k=13, compute_eigenvectors=False)
# args to scipy.sparse.linalg.eighs like can be passed with scipy_args={'tol':1e-8}
# [3., 4., 4., 4., 4., 4., 4., 5., 5., 5., 5., 5., 5.]

x_list = [(1 / np.sqrt(2))*(netket.operator.boson.create(hi_space,i)+netket.operator.boson.destroy(hi_space,i)) for i in range(N)]
### Quartic Interaction for bosons
V_b = (
    x_list[2] * x_list[2] * x_list[3] * x_list[3]
    + x_list[2] * x_list[2] * x_list[4] * x_list[4]
    + x_list[1] * x_list[1] * x_list[3] * x_list[3]
    + x_list[1] * x_list[1] * x_list[5] * x_list[5]
    + x_list[0] * x_list[0] * x_list[4] * x_list[4]
    + x_list[0] * x_list[0] * x_list[5] * x_list[5]
    - 2 * x_list[0] * x_list[2] * x_list[3] * x_list[5]
    - 2 * x_list[0] * x_list[1] * x_list[3] * x_list[4]
    - 2 * x_list[1] * x_list[2] * x_list[4] * x_list[5]
)

g2N = 0.2  # 't Hooft coupling lambda
H = H_free + (g2N / 2) * V_b

eig_vals = netket.exact.lanczos_ed(H, k=4, compute_eigenvectors=False, scipy_args={"tol": 1e-8})

class MeanFieldModel(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, x):
        gam = self.param("gamma", flax.linen.initializers.normal(), (1,), float)
        p = flax.linen.log_sigmoid(gam * x)
        ret = 0.5 * jnp.sum(p, axis=-1)
        return ret

model_mf = MeanFieldModel()
sampler = netket.sampler.MetropolisLocal(hi_space, n_chains=4)
# sampler = netket.sampler.ExactSampler(hi)
vstate = netket.vqs.MCState(sampler, model_mf, n_samples=2**9)
gs = netket.VMC(
    hamiltonian=H,
    optimizer=netket.optimizer.Sgd(learning_rate=0.05),
    preconditioner=netket.optimizer.SR(diag_shift=0.1),
    variational_state=vstate,
)
logger_mf = netket.logging.RuntimeLog()
gs.run(n_iter=300, out=logger_mf)
energy_mf = vstate.expect(H)


model = netket.models.Jastrow(kernel_init=flax.linen.initializers.normal())
vstate = netket.vqs.MCState(sampler, model, n_samples=2**10)
gs = netket.VMC(
    hamiltonian=H,
    optimizer=netket.optimizer.Sgd(learning_rate=0.05),
    preconditioner=netket.optimizer.SR(diag_shift=0.1),
    variational_state=vstate,
)
def hf_accept_callback(step, logged_data, driver):
    logged_data["acceptance"] = float(driver.state.sampler_state.acceptance)
    return True
logger_jastrow = netket.logging.RuntimeLog()
gs.run(n_iter=300, out=logger_jastrow, callback=hf_accept_callback)
energy_jastrow = vstate.expect(H)

iter_jastrow = logger_jastrow.data["Energy"].iters
energy_history_jastrow = logger_jastrow.data["Energy"].Mean.real
energy_error_history_jastrow = logger_jastrow.data["Energy"].Sigma.real
logger_jastrow.data["acceptance"]['value']

fig, ax = plt.subplots(figsize=(10, 5))
ax.errorbar(iter_jastrow, energy_history_jastrow, yerr=energy_error_history_jastrow, label='Jastrow')
ax.axhline(eig_vals[0], label="Exact")
ax.legend()
ax.set_xlabel("Iterations")
ax.set_ylabel("Energy (Re)")
fig.tight_layout()
fig.savefig('tbd00.png')
