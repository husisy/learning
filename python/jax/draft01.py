import numpy as np
import jax

import jax.numpy as jnp

hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

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
jnp1 = jnp0.at[0].set(10) #not effect jnp0
jnp2 = jnp0.at[[1],[1,2]].set(10)
jnp3 = jax.ops.index_update(jnp0, jax.ops.index[1], 10)
jnp4 = jax.ops.index_add(jnp0, jax.ops.index[:1, 1:], 10)


## jit
hf0 = lambda x: x.reshape(np.prod(x.shape))
hf1 = lambda x: x.reshape(jnp.array(x.shape).prod())
hf0_jit = jax.jit(hf0)
# hf1_jax = jax.jit(hf1) #fail, jnp.array() will be traced, but reshape(xxx) shape be static
jnp0 = jax.random.normal(jnp_rng(), (3,2), dtype=jnp.float32)
jnp1 = hf0_jit(jnp0)


# import numpy as np
# from jax import grad, jit
# from jax import lax
# from jax import random
# import jax
# import jax.numpy as jnp
# import matplotlib as mpl
# from matplotlib import pyplot as plt
# from matplotlib import rcParams
# rcParams['image.interpolation'] = 'nearest'
# rcParams['image.cmap'] = 'viridis'
# rcParams['axes.grid'] = False


## grad
hf0 = lambda x: jnp.sum(x**2 + x**3)
hf0_jit = jax.jit(hf0)
hf0_jit_grad = jax.grad(hf0_jit)
jnp0 = jax.random.normal(jnp_rng(), (3,2), dtype=jnp.float32)
ret_ = np.array(2*jnp0+3*jnp0**2)
ret0 = hf0_jit_grad(jnp0)
assert hfe(ret_, np.array(ret0)) < 1e-5

hf0 = lambda x,y: jnp.sum(x**2 + y**3)
hf0_jit = jax.jit(hf0)
hf0_jit_grad = jax.grad(hf0_jit, argnums=(0,1))
jnp0 = jax.random.normal(jnp_rng(), (3,2), dtype=jnp.float32)
jnp1 = jax.random.normal(jnp_rng(), (3,2), dtype=jnp.float32)
ret0,ret1 = hf0_jit_grad(jnp0, jnp1)
ret0_ = 2*np.array(jnp0)
ret1_ = 3*np.array(jnp1)**2
assert hfe(ret0_, np.array(ret0)) < 1e-5
assert hfe(ret1_, np.array(ret1)) < 1e-5
# jax.value_and_grad

hf0 = lambda x: jnp.sum(x**2)**2
hf0_grad = jax.grad(hf0)
hf0_grad_grad = jax.jacfwd(hf0_grad)
jnp0 = jax.random.normal(jnp_rng(), (3,), dtype=jnp.float32)
np0 = np.array(jnp0)
ret0 = hf0_grad(jnp0)
ret1 = hf0_grad_grad(jnp0)
ret0_ = 4*np0*(np0**2).sum()
ret1_ = 4*(np0**2).sum()*np.eye(np0.shape[0]) + 8*np0*np0[:,np.newaxis]
assert hfe(ret0_, np.array(ret0)) < 1e-5
assert hfe(ret1_, np.array(ret1)) < 1e-5
