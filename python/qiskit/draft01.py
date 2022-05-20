import numpy as np

import qiskit
import qiskit.providers.aer

aer_qasm_sim = qiskit.providers.aer.QasmSimulator()
hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


# from qiskit.providers.aer.noise import NoiseModel
# from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error



def get_noise(p_measure, p_gate):
    error_meas = qiskit.providers.aer.noise.errors.pauli_error([('X',p_measure), ('I',1-p_measure)])
    error_gate1 = qiskit.providers.aer.noise.errors.depolarizing_error(p_gate, 1)
    error_gate2 = error_gate1.tensor(error_gate1)

    noise_model = qiskit.providers.aer.noise.NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_meas, "measure") # measurement error is applied to measurements
    noise_model.add_all_qubit_quantum_error(error_gate1, ["x"]) # single qubit gate error is applied to x gates
    noise_model.add_all_qubit_quantum_error(error_gate2, ["cx"]) # two qubit gate error is applied to cx gates
    return noise_model


noise_model = get_noise(0.01,0.01)

qc0 = qiskit.QuantumCircuit(3, 3, name='0')
qc0.measure(qc0.qregs[0],qc0.cregs[0])
counts = qiskit.execute(qc0, aer_qasm_sim, noise_model=noise_model, shots=1000).result().get_counts()
print(counts)

qc1 = qiskit.QuantumCircuit(3, 3, name='0')
qc1.x(qc1.qregs[0])
qc1.measure(qc1.qregs[0], qc1.cregs[0])
counts = qiskit.execute(qc1, aer_qasm_sim, noise_model=noise_model).result().get_counts()
print(counts)

# fully random state
noise_model = get_noise(0.5,0.0)
counts = qiskit.execute(qc1, aer_qasm_sim, noise_model=noise_model).result().get_counts()
print(counts)

import qiskit.ignis.verification
#TODO qiskit.ignis is deprecated, to be replaced by qiskit-experiments project

qiskit.ignis.verification.topological_codes.RepetitionCode
qiskit.ignis.verification.topological_codes.lookuptable_decoding
qiskit.ignis.verification.topological_codes.GraphDecoder

n = 3
T = 1
code = qiskit.ignis.verification.topological_codes.RepetitionCode(n, T)
# code.circuit['0'].draw()
# code.circuit['1'].draw()


def get_raw_results(code, noise_model=None):
    circuits = code.get_circuit_list()
    raw_results = {}
    for log in range(2):
        job = qiskit.execute( circuits[log], Aer.get_backend('qasm_simulator'), noise_model=noise_model)
        raw_results[str(log)] = job.result().get_counts(str(log))
    return raw_results

raw_results = get_raw_results(code)
for log in raw_results:
    print('Logical',log,':',raw_results[log],'\n')
