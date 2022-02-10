import numpy as np
import jax

import jax.numpy as jnp


hfe = lambda x,y,eps=1e-5:np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

def generate_jnp_rng(seed):
    # not functional pure, just used for demo, use jax.random.split in regular code
    state = [jax.random.PRNGKey(seed)]
    def hf0(num0=None):
        if num0 is None:
            state[0], ret = jax.random.split(state[0])
        else:
            state[0], *ret = jax.random.split(state[0], num0)
        return ret
    return hf0

def test_random():
    N0 = 3
    N1 = 5
    jnp_prngkey = jax.random.PRNGKey(0)
    z0 = np.stack([np.array(jax.random.normal(jnp_prngkey, (N0,), dtype=jnp.float32)) for _ in range(N1)])
    assert np.max(np.std(z0, axis=0)) < 1e-6

    jnp_rng = generate_jnp_rng(0)
    z1 = np.stack([np.array(jax.random.normal(jnp_rng(), (N0,), dtype=jnp.float32)) for _ in range(N1)])
    assert np.min(np.std(z1, axis=0)) > 1e-3
    z2 = np.stack([np.array(jax.random.normal(x, (N0,), dtype=jnp.float32)) for x in jnp_rng(N1)])
    assert np.min(np.std(z2, axis=0)) > 1e-3
