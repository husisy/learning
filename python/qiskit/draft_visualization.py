import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import qiskit
import qiskit.visualization
import qiskit.providers.aer

aer_qasm_sim = qiskit.providers.aer.QasmSimulator()
aer_state_sim = qiskit.providers.aer.StatevectorSimulator()

## basic
qc0 = qiskit.QuantumCircuit(2, 2)
qc0.h(0)
qc0.cx(0, 1)
qc0.measure([0,1], [0,1])
# qc0.draw() #text mpl latex latex_source
# qc0.draw('mpl') #import matplotlib first
# qiskit.visualization.circuit_drawer
qc0_compiled = qiskit.transpile(qc0, aer_qasm_sim)
result0 = aer_qasm_sim.run(qc0_compiled, shots=1000).result()
count0 = result0.get_counts(qc0)
result1 = aer_qasm_sim.run(qc0_compiled, shots=1000).result()
count1 = result1.get_counts(qc0)
# qiskit.visualization.plot_histogram(count0)
# qiskit.visualization.plot_histogram([count0, count1], legend=['run0','run1'])

qc0 = qiskit.QuantumCircuit(2, 2)
qc0.h(0)
qc0.cx(0, 1) #no measure
qc0_compiled = qiskit.transpile(qc0, aer_state_sim)
result = aer_state_sim.run(qc0_compiled).result()
psi  = result.get_statevector(qc0)
# qiskit.visualization.plot_state_city(psi)
# qiskit.visualization.plot_state_hinton(psi)
# qiskit.visualization.plot_state_paulivec(psi)
# qiskit.visualization.plot_bloch_multivector(psi)


## visulization
q_a = qiskit.QuantumRegister(3, name='qa')
q_b = qiskit.QuantumRegister(5, name='qb')
c_a = qiskit.ClassicalRegister(3)
c_b = qiskit.ClassicalRegister(5)
qc0 = qiskit.QuantumCircuit(q_a, q_b, c_a, c_b)
qc0.x(q_a[1])
qc0.x(q_b[1])
qc0.x(q_b[2])
qc0.x(q_b[4])
qc0.barrier()
qc0.h(q_a)
qc0.barrier(q_a)
qc0.h(q_b)
qc0.cswap(q_b[0], q_b[1], q_b[2])
qc0.cswap(q_b[2], q_b[3], q_b[4])
qc0.cswap(q_b[3], q_b[4], q_b[0])
qc0.barrier(q_b)
qc0.measure(q_a, c_a)
qc0.measure(q_b, c_b)
qc0.draw(output='mpl', reverse_bits=True)


## dag
q = qiskit.QuantumRegister(3, 'q')
c = qiskit.ClassicalRegister(3, 'c')
qc0 = qiskit.QuantumCircuit(q, c)
qc0.h(q[0])
qc0.cx(q[0], q[1])
qc0.measure(q[0], c[0])
qc0.rz(0.5, q[1]).c_if(c, 2)

dag = qiskit.converters.circuit_to_dag(qc0)
# qiskit.visualization.dag_drawer(dag)
nodei = dag.op_nodes()[3]
nodei.name
nodei.op
nodei.op.condition
nodei.qargs
nodei.cargs

# add op
dag.apply_operation_back(qiskit.circuit.library.HGate(), qargs=[q[0]])
dag.apply_operation_front(qiskit.circuit.library.CCXGate(), qargs=[q[0], q[1], q[2]], cargs=[])

# substitute op
p = qiskit.QuantumRegister(2, "p")
op_new = qiskit.dagcircuit.DAGCircuit()
op_new.add_qreg(p)
op_new.apply_operation_back(qiskit.circuit.library.CHGate(), qargs=[p[1], p[0]])
op_new.apply_operation_back(qiskit.circuit.library.U2Gate(0.1, 0.2), qargs=[p[1]])
op_old = dag.op_nodes(op=qiskit.circuit.library.CXGate)[0]
dag.substitute_node_with_dag(node=op_old, input_dag=op_new, wires=[p[0], p[1]])
qc1 = qiskit.converters.dag_to_circuit(dag)


# unitary
np0 = np.array([[1,1],[1,-1]])/np.sqrt(2)
qiskit.visualization.array_to_latex(np0) #in jupyter
