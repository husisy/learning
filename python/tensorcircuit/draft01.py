import numpy as np
import tensorflow as tf

import tensorcircuit as tc

K = tc.set_backend("tensorflow")

np_rng = np.random.default_rng()
hf_randc = lambda *size: np_rng.normal(size=size) + 1j*np_rng.normal(size=size)

@tf.function
def hf_dummy_circuit(theta):
    num_qubit = theta.shape[1]
    circ = tc.Circuit(num_qubit)
    with tf.GradientTape() as tape:
        tape.watch(theta)
        for i in range(num_qubit):
            circ.rx(i, theta=theta[0, i])
            circ.rz(i, theta=theta[1, i])
        for i in range(0, num_qubit-1, 2):
            circ.cnot(i, i+1)
        loss = tf.math.real(sum(circ.expectation([tc.gates.z(), [i]]) for i in range(num_qubit)))
    grad = tape.gradient(loss, theta)
    return loss,grad

num_qubit = 2
tf0 = tf.random.normal([2, num_qubit])
fval,grad = hf_dummy_circuit(tf0)



# circuit intermediate representation
circ = tc.Circuit(2)
circ.cnot(0, 1)
circ.crx(1, 0, theta=0.2)
circ.to_qir()


## MPS and MPO
n = 3
nodes = [tc.gates.Gate(np.array([0.0, 1.0])) for _ in range(n)]
mps = tc.quantum.QuVector([x[0] for x in nodes])
c = tc.Circuit(n, mps_inputs=mps)
c.x(0)
c.expectation_ps(z=[0])
# 1.0

hf0 = lambda x: x/np.linalg.norm(x, ord=2)
z0 = [hf0(hf_randc(2)) for _ in range(3)]
z1 = tc.quantum.QuVector([tc.gates.Gate(x)[0] for x in z0])
z2 = z1.eval()
tmp0 = np.einsum(z0[0], [0], z0[1], [1], z0[2], [2], [0, 1, 2], optimize=True).reshape(-1)
assert np.abs(tmp0 - z2.reshape(-1)).max() < 1e-10
