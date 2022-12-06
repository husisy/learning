import numpy as np
import jax
import jax.numpy as jnp
import functools

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

# jax.random
# jax.grad
# jax.jit
# jax.vmap

# TODO seed=None
def generate_jnp_rng(seed=None):
    if seed is None:
        seed = int(np.random.default_rng().integers(int(1e18)))
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
assert np.abs(np.asarray(jnp1)-np.asarray(jnp1_)).max() < 1e-5

hf1 = jax.grad(jax.jit(jax.grad(jax.jit(jax.grad(hf0)))))
hf1(1.0)

hf0_hessian = jax.jit(jax.jacfwd(jax.jacrev(hf0)))
jnp2 = hf0_hessian(jnp0)


# auto-vectorization with vmap
mat = jax.random.normal(jnp_rng(), (150, 100))
batched_x = jax.random.normal(jnp_rng(), (10, 100))
hf_mat_vec = lambda v: jnp.dot(mat, v)
# hf_mat_vec_batch = lambda v_batch: jnp.stack([hf_mat_vec(v) for v in v_batch])
hf_mat_vec_batch = lambda v_batch: jnp.dot(v_batch, mat.T)
#%timeit hf_mat_vec_batch(batched_x).block_until_ready()
# 138 µs ± 2.93 µs per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
hf_mat_vec_batch_jit = jax.jit(hf_mat_vec_batch)
#%timeit hf_mat_vec_batch_jit(batched_x).block_until_ready()
# 24.9 µs ± 153 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
hf_mat_vec_vmap = jax.jit(lambda v_batch: jax.vmap(hf_mat_vec)(v_batch))
#%timeit hf_mat_vec_vmap(batched_x).block_until_ready()
# 24.7 µs ± 145 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)


# lax
jnp.add(1, 1.0)
# jax.lax.add(1, 1.0) #fail TypeError
jax.lax.add(1.0, 1.0)


# jit
hf0 = lambda x: (x-x.mean(0)) / x.std(0)
hf0_jit = jax.jit(hf0)
jnp0 = jax.random.normal(jnp_rng(), (100,2), dtype=jnp.float32)
jnp1 = hf0_jit(jnp0)
tmp0 = jax.make_jaxpr(hf0)(jnp0)

@jax.jit
def demo_jax_cond(jnp0):
    hf_true = lambda x: x
    hf_false = lambda x: -x
    ret = jax.lax.cond(jnp0.sum()>0, hf_true, hf_false, jnp0)
    return ret
jnp0 = jax.random.normal(jnp_rng(), (3,2), dtype=jnp.float32)
jnp1 = demo_jax_cond(jnp0)



hf0 = lambda x,neg: (-x if neg else x)
hf0_jit = functools.partial(jax.jit, static_argnums=(1,))(hf0)
hf0_jit(1, True)
hf0_jit(-1, False) #compile twice


# static vs traced operation
hf0_jnp_jit = jax.jit(lambda x: x.reshape(jnp.array(x.shape).prod()))
hf0_np_jit = jax.jit(lambda x: x.reshape(np.prod(x.shape)))
jnp0 = jax.random.normal(jnp_rng(), (2,3), dtype=jnp.float32)
# hf0_jnp_jit(jnp0) #error
hf0_np_jit(jnp0)
