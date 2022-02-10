import os
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import qiskit
import qiskit.providers.aer
import qiskit.providers.ibmq
import qiskit.tools.visualization

with open(os.path.expanduser('~/qiskit_token.txt'), 'r') as fid:
    IBMQ_TOKEN = fid.read().strip()

ibmq_provider = qiskit.providers.ibmq.IBMQ.enable_account(IBMQ_TOKEN, group='open', hub='ibm-q', project='main')
# ibmq_provider = qiskit.providers.ibmq.IBMQ.enable_account(IBMQ_TOKEN, group='qscitech-quantum', hub='ibm-q-education', project='qc-bc-workshop')
aer_qasm_sim = qiskit.providers.aer.QasmSimulator()

theta = qiskit.circuit.Parameter('θ')
qc0 = qiskit.QuantumCircuit(2, 1)
qc0.h(0)
qc0.cx(0, 1)
qc0.rz(theta, 1)
qc0.cx(0, 1)
qc0.h(0)
qc0.measure([0], [0])
theta_np = np.linspace(0, 2*np.pi, 32)
circuits = [qc0.bind_parameters({theta: x}) for x in theta_np]

result = aer_qasm_sim.run(qiskit.transpile(circuits, aer_qasm_sim)).result()
count_sim = np.array([(x.get('0',0)) for x in result.get_counts()])

# result = ibmq_provider.run_circuits(qc0.bind_parameters({theta: 0.1}), backend_name='ibmq_lima', optimization_level=3).result()
# count_qc = result.get_counts().get('0')
result = ibmq_provider.run_circuits(circuits, backend_name='ibmq_lima', optimization_level=3).result()
count_qc = np.array([x.get('0',0) for x in result.get_counts()])
# [1008,951,933,849,817,731,695,597,523,439,331,264,193,166,117, 94, 85, 88,123,190,225,280,366,467,578,657,738,817,879,920,951,1016]


fig,(ax0,ax1) = plt.subplots(1, 2, figsize=(8,4))
qiskit.tools.visualization.circuit_drawer(qc0, output='mpl', ax=ax0)
ax1.plot(theta_np, count_sim, label='aer.QasmSimulator')
ax1.plot(theta_np, count_qc, label='ibmq_lima')
ax1.legend()
ax1.set_xlabel(r'$\theta$')
ax1.set_ylabel('measure(q0)')
fig.tight_layout()
