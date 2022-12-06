import numpy as np
import tensorflow as tf
import tensorcircuit as tc

K = tc.set_backend("tensorflow") #tensorflow torch jax
tc.set_dtype("complex128")
tc.set_contractor("greedy")

circ = tc.Circuit(2)
circ.H(0)
circ.CNOT(0,1)
circ.rx(1, theta=0.2)


print(circ.wavefunction())
print(circ.expectation_ps(z=[0, 1]))
print(circ.sample())

# TODO test with numpysim


n = 1


def loss(params, n):
    c = tc.Circuit(n)
    for i in range(n):
        c.rx(i, theta=params[0, i])
    for i in range(n):
        c.rz(i, theta=params[1, i])
    loss = 0.0
    for i in range(n):
        loss += c.expectation([tc.gates.z(), [i]])
    return K.real(loss)


vgf = K.jit(K.value_and_grad(loss), static_argnums=1)
params = K.implicit_randn([2, n])
print(vgf(params, n))  # get the quantum loss and the gradient


def loss(params, n):
    c = tc.Circuit(n)
    for i in range(n):
        c.rx(i, theta=params[0, i])
    for i in range(n):
        c.rz(i, theta=params[1, i])
    loss = 0.0
    for i in range(n):
        loss += c.expectation([tc.gates.z(), [i]])
    return tf.math.real(loss)

def vgf(params, n):
    with tf.GradientTape() as tape:
        tape.watch(params)
        l = loss(params, n)
    return l, tape.gradient(l, params)

vgf = tf.function(vgf)
params = tf.random.normal([2, n])
print(vgf(params, n))  # get the quantum loss and the gradient
