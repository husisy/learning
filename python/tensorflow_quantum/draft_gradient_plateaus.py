# https://github.com/tensorflow/quantum/blob/master/docs/tutorials/barren_plateaus.ipynb
# https://www.nature.com/articles/s41467-018-07090-4
import cirq
import sympy
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.ion()

import tensorflow_quantum as tfq

def generate_random_qnn(qubits, symbol, depth):
    circuit = cirq.Circuit()
    for x in qubits:
        circuit += cirq.ry(np.pi/4)(x)
    tmp0 = [cirq.rx, cirq.ry, cirq.rz]
    for d in range(depth):
        for i, qubit in enumerate(qubits):
            random_rot = symbol if ((i==0) and (d==0)) else np.random.uniform()*2*np.pi
            circuit += tmp0[np.random.randint(0, 3)](random_rot)(qubit)
        for src, dest in zip(qubits, qubits[1:]):
            circuit += cirq.CZ(src, dest)
    return circuit


def process_batch(circuits, symbol, op):
    expectation = tfq.layers.Expectation()
    circuit_tensor = tfq.convert_to_tensor(circuits)
    values_tensor = tf.convert_to_tensor(np.random.uniform(0, 2 * np.pi, (n_circuits, 1)).astype(np.float32))
    with tf.GradientTape() as g:
        g.watch(values_tensor)
        forward = expectation(circuit_tensor, operators=op, symbol_names=[symbol], symbol_values=values_tensor)
    grads = g.gradient(forward, values_tensor).numpy().reshape(-1)
    return grads


def generate_identity_qnn(qubits, symbol, block_depth, total_depth):
    """Generate random QNN's with the same structure from Grant et al."""
    circuit = cirq.Circuit()
    prep_and_U = generate_random_qnn(qubits, symbol, block_depth)
    circuit += prep_and_U
    circuit += cirq.resolve_parameters((prep_and_U[1:])**-1, param_resolver={symbol: np.random.uniform()*2*np.pi})
    for d in range(total_depth - 1):
        U_circuit = generate_random_qnn(qubits, np.random.uniform()*2*np.pi, block_depth)[1:]
        circuit += U_circuit
        circuit += U_circuit**-1
    return circuit


num_qubit_list = [2*x for x in range(2, 7)]  # Ranges studied in paper are between 2 and 24.
depth = 50  # Ranges studied in paper are between 50 and 500.
n_circuits = 200
theta_grad = []
for num_qubit in tqdm(num_qubit_list): #90 seconds
    qubits = cirq.GridQubit.rect(1, num_qubit)
    symbol = sympy.Symbol('theta')
    circuits = [generate_random_qnn(qubits, symbol, depth) for _ in range(n_circuits)]
    op = cirq.Z(qubits[0]) * cirq.Z(qubits[1])
    theta_grad.append(process_batch(circuits, symbol, op))
theta_grad = np.stack(theta_grad)

fig,ax = plt.subplots()
ax.semilogy(num_qubit_list, theta_grad.std(axis=1))
ax.set_title('Gradient Variance in QNNs')
ax.set_xlabel('n_qubits')
ax.set_xticks(num_qubit_list)
ax.set_ylabel('$\\partial \\theta$ variance')


block_depth = 10
total_depth = 5
heuristic_theta_grad = []
for num_qubit in tqdm(num_qubit_list): #160 seconds
    # Generate the identity block circuits and observable for the given num_qubit
    qubits = cirq.GridQubit.rect(1, num_qubit)
    symbol = sympy.Symbol('theta')
    circuits = [generate_identity_qnn(qubits, symbol, block_depth, total_depth) for _ in range(n_circuits)]
    op = cirq.Z(qubits[0]) * cirq.Z(qubits[1])
    heuristic_theta_grad.append(process_batch(circuits, symbol, op))
heuristic_theta_grad = np.stack(heuristic_theta_grad)

fig,ax = plt.subplots()
ax.semilogy(num_qubit_list, theta_grad.std(axis=1))
ax.semilogy(num_qubit_list, heuristic_theta_grad.std(axis=1))
ax.set_title('Heuristic vs. Random')
ax.set_xlabel('n_qubits')
ax.set_xticks(num_qubit_list)
ax.set_ylabel('$\\partial \\theta$ variance')
