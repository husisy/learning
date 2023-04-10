import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import qiskit
import qiskit.providers.aer
import qiskit.visualization
# import qiskit.test.mock

aer_sim = qiskit.Aer.get_backend('aer_simulator')
aer_qasm_sim = qiskit.providers.aer.QasmSimulator()
aer_state_sim = qiskit.providers.aer.StatevectorSimulator()
aer_unitary_sim = qiskit.providers.aer.UnitarySimulator()
# backend = qiskit.Aer.get_backend('statevector_simulator')
# backend = qiskit.Aer.get_backend('unitary_simulator')

np_rng = np.random.default_rng()
hf_randc = lambda *size: np_rng.normal(size=size) + 1j*np_rng.normal(size=size)

##
qc0 = qiskit.QuantumCircuit(2, 2) #quantum_register, classical_register
qc0.h(0)
qc0.cx(0, 1) #control(0) target(1)
qc0.measure([0,1], [0,1])
# compile the qc0 down to low-level QASM instructions supported by the backend
qc0_compiled = qiskit.transpile(qc0, aer_qasm_sim)
result = aer_qasm_sim.run(qc0_compiled, shots=1000).result()
counts = result.get_counts(qc0_compiled) #str->int #c1c0
# qc0.draw('mpl') #initial_state=True


## demo compose operation (below two are equivalent)
qc0 = qiskit.QuantumCircuit(3, 3)
qc0.h(0)
qc0.cx(0, 1)
qc0.cx(0, 2)
qc0.barrier([0,1,2])
qc0.measure([0,1,2], [0,1,2])

qc0 = qiskit.QuantumCircuit(3, 3)
qc0.h(0)
qc0.cx(0, 1)
qc0.cx(0, 2)
qc1 = qiskit.QuantumCircuit(3, 3)
qc1.barrier([0,1,2])
qc1.measure([0,1,2], [0,1,2])
qc2 = qc1.compose(qc0, range(3), front=True) #qc0 and qc1 are not changed


##
qc0 = qiskit.QuantumCircuit(3, 3)
qc0.h(0)
qc0.cx(0, 1)
qc0.cx(0, 2)
result = aer_state_sim.run(qc0).result()
statevec = result.get_statevector(qc0)
statevec.data
# qiskit.visualization.plot_state_city(statevec)
result = aer_unitary_sim.run(qc0).result()
operator = result.get_unitary(qc0)
operator.data

## https://qiskit.org/documentation/tutorials/circuits/01_circuit_basics.html
qc0 = qiskit.QuantumCircuit(3)
qc0.h(0)
qc0.cx(0, 1)
qc0.cx(0, 2)
state = qiskit.quantum_info.Statevector.from_int(i=0, dims=2**3)
state = state.evolve(qc0) #10000001
# qiskit.visualization.array_to_latex(state)
# state.draw('qsphere')
# state.draw('hinton')
U = qiskit.quantum_info.Operator(qc0)
U.data


## classical condition op
q = qiskit.QuantumRegister(1)
c = qiskit.ClassicalRegister(1)
qc0 = qiskit.QuantumCircuit(q, c)
qc0.x(q[0]).c_if(c, 0) #InstructionSet
qc0.measure(q,c)
# qc0.draw()


##
my_gate = qiskit.circuit.Gate(name='my_gate', num_qubits=2, params=[])
qr = qiskit.QuantumRegister(size=3, name='q')
circ = qiskit.QuantumCircuit(qr)
circ.append(my_gate, [qr[0], qr[1]])
circ.append(my_gate, [qr[1], qr[2]])

## composite gates
sub_q = qiskit.QuantumRegister(2)
sub_circ = qiskit.QuantumCircuit(sub_q, name='sub_circ')
sub_circ.h(sub_q[0])
sub_circ.crz(1, sub_q[0], sub_q[1])
sub_circ.barrier()
sub_circ.id(sub_q[1])
sub_circ.u(1, 2, -2, sub_q[0])
sub_inst = sub_circ.to_instruction() # Convert to a gate and stick it into an arbitrary place in the bigger circuit

qr = qiskit.QuantumRegister(3, 'q')
circ = qiskit.QuantumCircuit(qr)
circ.h(qr[0])
circ.cx(qr[0], qr[1])
circ.cx(qr[1], qr[2])
circ.append(sub_inst, [qr[1], qr[2]])
decomposed_circ = circ.decompose() #out-of-place operation
# decomposed_circ.draw()


## Parameterized circuits
theta = qiskit.circuit.Parameter('θ')
n = 5
qc = qiskit.QuantumCircuit(5, 1)
qc.h(0)
for i in range(n-1):
    qc.cx(i, i+1)
qc.barrier()
qc.rz(theta, range(5))
qc.barrier()
for i in reversed(range(n-1)):
    qc.cx(i, i+1)
qc.h(0)
qc.measure(0, 0)
qc.parameters
theta_range = np.linspace(0, 2 * np.pi, 128)
circuits = [qc.bind_parameters({theta: x}) for x in theta_range]
counts = aer_qasm_sim.run(qiskit.transpile(circuits, aer_qasm_sim)).result().get_counts()
count_01 = np.array([(x.get('0',0),x.get('1',0)) for x in counts])
# fig,ax = plt.subplots()
# ax.plot(theta_range, count_01[:,0], '.-', label='0')
# ax.plot(theta_range, count_01[:,1], '.-', label='1')
# ax.legend()

## reducing compilation cost
theta = qiskit.circuit.Parameter('θ')
n = 5
qc = qiskit.QuantumCircuit(5, 1)
qc.h(0)
for i in range(n-1):
    qc.cx(i, i+1)
qc.barrier()
qc.rz(theta, range(5))
qc.barrier()
for i in reversed(range(n-1)):
    qc.cx(i, i+1)
qc.h(0)
qc.measure(0, 0)
qc.parameters
theta_range = np.linspace(0, 2 * np.pi, 128)
transpiled_qc = qiskit.transpile(qc, backend=qiskit.test.mock.FakeVigo())
tmp0 = [transpiled_qc.bind_parameters({theta:x}) for x in theta_range]
qobj = qiskit.compiler.assemble(tmp0, backend=qiskit.test.mock.FakeVigo())
counts = aer_qasm_sim.run(qobj).result().get_counts()


## half adder https://qiskit.org/textbook/ch-states/atoms-computation.html
for bitstr in ['00','01','10','11']:
    circ0 = qiskit.QuantumCircuit(4,2)
    if bitstr[0]=='1':
        circ0.x(0)
    if bitstr[1]=='1':
        circ0.x(1)
    circ0.barrier()
    circ0.cx(0,2)
    circ0.cx(1,2)
    circ0.ccx(0,1,3)
    circ0.barrier()
    circ0.measure(2,0)
    circ0.measure(3,1)

    qc0_compiled = qiskit.transpile(circ0, aer_qasm_sim)
    result = aer_qasm_sim.run(qc0_compiled, shots=1000).result()
    counts = result.get_counts(qc0_compiled) #str->int
    print(bitstr, counts)


## get state vector
def get_circuit_statevector(circ0):
    circ0 = circ0.copy()
    circ0.save_statevector()
    qobj = qiskit.assemble(circ0)
    ret = aer_sim.run(qobj).result().get_statevector().data
    return ret
num_qubit = 2
tmp0 = hf_randc(2**num_qubit)
np0 = tmp0 / np.linalg.norm(tmp0)
circ0 = qiskit.QuantumCircuit(num_qubit)
circ0.initialize(np0, list(range(num_qubit)))
np1 = get_circuit_statevector(circ0)

## get unitary
def get_circuit_unitary(circ0):
    circ0 = circ0.copy()
    circ0.save_unitary()
    qobj = qiskit.assemble(circ0)
    ret = aer_sim.run(qobj).result().get_unitary().data
    return ret
circ0 = qiskit.QuantumCircuit(2)
circ0.x(1)
np0 = get_circuit_unitary(circ0)


circ0 = qiskit.QuantumCircuit(2) #SWAP
circ0.cnot(0, 1)
circ0.cnot(1, 0)
circ0.cnot(0, 1)
get_circuit_unitary(circ0)


circ0 = qiskit.QuantumCircuit(2) #close to Toffoli
circ0.ch(0, 2)
circ0.cz(1, 2)
circ0.ch(0, 2)
# get_circuit_unitary(circ0) #fail
