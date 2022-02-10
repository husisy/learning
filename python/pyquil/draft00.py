import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter
plt.ion()

from pyquil import Program, get_qc, list_quantum_computers
from pyquil.gates import CNOT, H, X, Z, I, RY, MEASURE
from pyquil.paulis import sZ

from utils import setup_qvm_quilc_connection

FC, WF_SIM = setup_qvm_quilc_connection()


WF_SIM.wavefunction(Program(X(0))).amplitudes
WF_SIM.wavefunction(Program(H(0))).amplitudes
WF_SIM.wavefunction(Program(H(0), CNOT(control=0,target=1))).amplitudes

QC = get_qc('2q-qvm', connection=FC)
circuit = Program(H(0), CNOT(0,1))
print('H-CNOT: ', WF_SIM.wavefunction(circuit))
print('H-CNOT measure: ', QC.run_and_measure(circuit, trials=10))


# QC = get_qc('1q-qvm', connection=FC)
QC = get_qc('1q-noisy-qvm', connection=FC)
circuit = Program()
ro = circuit.declare('ro', memory_type='BIT', memory_size=1)
circuit += MEASURE(0, ro[0])
circuit.wrap_in_numshots_loop(100)
executable = QC.compile(circuit)
result = QC.run(executable)[:,0]
print('P(result=0): ', np.mean(result==0))
print('P(result=1): ', np.mean(result==1))


observable_list = [
    (1-sZ(0))*0.5, #expection of the first qubit
    (1-sZ(1))*0.5, #expection of the second qubit
    (1-sZ(0)*sZ(1))*0.5, #xor
]
circuit = Program(H(0), CNOT(0, 1))
for x in observable_list:
    tmp0 = WF_SIM.expectation(prep_prog=circuit, pauli_terms=x)
    print('{}: {}'.format(x, tmp0))


QC = get_qc('1q-qvm', connection=FC)
theta = np.linspace(0, np.pi, 20)
bitstring = [QC.run_and_measure(Program(RY(x, 0)), trials=10000)[0].mean() for x in theta]
plt.plot(theta, bitstring, 'o-')


def generate_GHZ_circuit(qubits):
    program = Program()
    program += H(qubits[0])
    for q1, q2 in zip(qubits, qubits[1:]):
        program += CNOT(q1, q2)
    return program
num_qubit = 3
num_trial = 1000
QC = get_qc('3q-noisy-qvm')
GHZ_circuit = generate_GHZ_circuit(list(range(num_qubit)))
print('GHZ_circuit wave function: ', WF_SIM.wavefunction(GHZ_circuit))
bitstring = QC.run_and_measure(GHZ_circuit, trials=num_trial)
tmp0 = Counter(zip(*[bitstring[x] for x in range(num_qubit)]))
for key,value in sorted(tmp0.items(), key=lambda x: x[0]):
    print('P{}: {}'.format(key, value/num_trial))


list_quantum_computers()


QC = get_qc('9q-square-qvm')
nx.draw(QC.qubit_topology())


QC = get_qc('1q-qvm')
quil_step0 = QC.compiler.quil_to_native_quil(Program(RY(0.5, 0))) #Quil to native quil
quil_step1 = QC.compiler.native_quil_to_executable(quil_step0) #native quil to executable
print(quil_step0)
print(quil_step1)


hf_ansatz_circuit = lambda x: Program(RY(x, 0))

QC = get_qc('1q-qvm')
theta = np.linspace(0, 2*np.pi, 21)
result = [QC.run_and_measure(hf_ansatz_circuit(x), trials=1000)[0].mean() for x in theta]
plt.plot(theta, result, 'o-')

def hf_parametric_circuit():
    circuit = Program()
    theta = circuit.declare('theta', memory_type='REAL')
    ro = circuit.declare('ro', memory_type='BIT', memory_size=1)
    circuit += RY(theta, 0)
    circuit += MEASURE(0, ro[0])
    return circuit

parametric_circuit = hf_parametric_circuit()
parametric_circuit.wrap_in_numshots_loop(shots=1000)
executable = QC.compile(parametric_circuit)
result = [QC.run(executable, memory_map={'theta': [x]})[:,0].mean() for x in theta]
plt.plot(theta, result, 'o-')


