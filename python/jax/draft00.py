import numpy as np
import jax
import jax.numpy as jnp

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

# jax.random
# jax.grad
# jax.jit
# jax.vmap

def generate_jnp_rng(seed):
    state = [jax.random.PRNGKey(seed)]
    def hf0(num0=None):
        if num0 is None:
            state[0], ret = jax.random.split(state[0])
        else:
            state[0], *ret = jax.random.split(state[0], num0)
        return ret
    return hf0


jnp_rng = generate_jnp_rng(0)
np_rng = np.random.default_rng()


jnp0 = jax.random.normal(jnp_rng(), (2,3), dtype=jnp.float32)
np0 = np.array(jnp0)
jnp1 = jax.device_put(np0)


jnp0 = jax.random.normal(jnp_rng(), (3000, 3000), dtype=jnp.float32)
jnp1 = jnp.dot(jnp0, jnp0.T)
#%timeit jnp.dot(jnp0, jnp0.T).block_until_ready()
# 8.52 ms ± 8.12 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)


# jit
hf_selu = lambda x,alpha=1.67,lmbda=1.05: lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)
hf_selu_jit = jax.jit(hf_selu)
jnp0 = jax.random.normal(jnp_rng(), (1000000,))
#%timeit hf_selu(jnp0).block_until_ready()
# 939 µs ± 10.8 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
#%timeit hf_selu_jit(jnp0).block_until_ready()
# 57.8 µs ± 420 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)


# autograd
hf0 = lambda x: jnp.sum(1 / (1 + jnp.exp(-x)))
hf0_grad = jax.grad(hf0)
jnp0 = jax.random.normal(jnp_rng(), (3,))
jnp1 = hf0_grad(jnp0)
jnp1_ = jnp.exp(-jnp0) / (1+jnp.exp(-jnp0))**2

hf1 = jax.grad(jax.jit(jax.grad(jax.jit(jax.grad(hf0)))))
hf1(1.0)

hf0_hessian = jax.jit(jax.jacfwd(jax.jacrev(hf0)))
jnp2 = hf0_hessian(jnp0)


# auto-vectorization with vmap
mat = jax.random.normal(jnp_rng(), (150, 100))
batched_x = jax.random.normal(jnp_rng(), (10, 100))
def apply_matrix(v):
    return jnp.dot(mat, v)
def naively_batched_apply_matrix(v_batched):
    return jnp.stack([apply_matrix(v) for v in v_batched])
#%timeit naively_batched_apply_matrix(batched_x).block_until_ready()
# 4.75 ms ± 1.39 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

@jax.jit
def batched_apply_matrix(v_batched):
    return jnp.dot(v_batched, mat.T)
#%timeit batched_apply_matrix(batched_x).block_until_ready()
# 40.4 µs ± 1.26 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)

@jax.jit
def vmap_batched_apply_matrix(v_batched):
    return jax.vmap(apply_matrix)(v_batched)
#%timeit vmap_batched_apply_matrix(batched_x).block_until_ready()
# 42.9 µs ± 93.3 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
