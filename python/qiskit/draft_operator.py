import numpy as np
import matplotlib.pyplot as plt
plt.ion()

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

import qiskit


op0 = qiskit.quantum_info.operators.Operator(np.random.rand(4,4)) #could also be non-square
op1 = qiskit.quantum_info.operators.Operator([[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]])
op2 = 0.233*op0 - 1j*op1
op0.data
op0.dim
op0.input_dims()
op0.output_dims()
op0.is_unitary()

pauli_xx = qiskit.quantum_info.operators.Pauli(label='XX')
op0 = qiskit.quantum_info.operators.Operator(pauli_xx)

gate0 = qiskit.circuit.library.CXGate()
gate1 = qiskit.circuit.library.RXGate(np.pi / 2)
op0 = qiskit.quantum_info.operators.Operator(gate0)

qc0 = qiskit.QuantumCircuit(3)
qc0.h(0)
for j in range(1, qc0.qregs[0].size):
    qc0.cx(j-1, j)
op0 = qiskit.quantum_info.operators.Operator(qc0)

pauli_xx = qiskit.quantum_info.operators.Pauli(label='XX')
op0 = qiskit.quantum_info.operators.Operator(pauli_xx)
qc0 = qiskit.QuantumCircuit(2, 2)
qc0.append(op0, [0, 1])
qc0.measure([0,1], [0,1])
qc1 = qiskit.QuantumCircuit(2, 2) #equivalent
qc1.append(pauli_xx, [0, 1])
qc1.measure([0,1], [0,1])


op0 = qiskit.quantum_info.operators.Operator(qiskit.quantum_info.operators.Pauli(label='X'))
op1 = qiskit.quantum_info.operators.Operator(qiskit.quantum_info.operators.Pauli(label='Z'))
op2 = op0.tensor(op1) #direct product A x B
assert hfe(op2.data, np.kron(op0.data, op1.data)) < 1e-7
op3 = op0.expand(op1) #direct product B x A
assert hfe(op3.data, np.kron(op1.data, op0.data)) < 1e-7
op4 = op0.compose(op1) #B @ A
assert hfe(op4.data, op1.data @ op0.data) < 1e-7


op0 = qiskit.quantum_info.operators.Operator(qiskit.circuit.library.XGate())
op1 = np.exp(2.3j) * op0
fidelity = qiskit.quantum_info.process_fidelity(op0, op1) #almost 1
