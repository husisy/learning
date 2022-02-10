import numpy as np
import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
plt.ion()

def forward(params, x):
    for layer in params[:-1]:
        x = jax.nn.relu(x @ layer['weights'] + layer['biases'])
    ret = x @ params[-1]['weights'] + params[-1]['biases']
    return ret

def loss_fn(params, x, y):
    return jnp.mean((forward(params, x) - y) ** 2)

LEARNING_RATE = 0.0001

@jax.jit
def update(params, x, y):
    grads = jax.grad(loss_fn)(params, x, y)
    ret = jax.tree_multimap(lambda p, g: p - LEARNING_RATE*g, params, grads)
    return ret

np_rng = np.random.default_rng()

layer_width = [1, 128, 128, 1]
tmp0 = [np_rng.normal(scale=np.sqrt(2/x),size=(x,y)) for x,y in zip(layer_width[:-1],layer_width[1:])]
tmp1 = [np.ones(x) for x in layer_width[1:]]
params = [{'weights':x, 'biases':y} for x,y in zip(tmp0,tmp1)]

xs = np.random.normal(size=(128, 1))
ys = xs ** 2

for _ in range(1000):
    params = update(params, xs, ys)

fig,ax = plt.subplots()
ax.scatter(xs, ys)
ax.scatter(xs, forward(params, xs), label='Model prediction')
ax.legend()
