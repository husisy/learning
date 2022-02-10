import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import qiskit
import qiskit.opflow
import qiskit.providers.aer
import qiskit_machine_learning.neural_networks

np_rng = np.random.default_rng(233)

aer_qasm_sim = qiskit.providers.aer.QasmSimulator()
aer_state_sim = qiskit.providers.aer.StatevectorSimulator()
qi_sv = qiskit.utils.QuantumInstance(aer_state_sim)
qi_qasm = qiskit.utils.QuantumInstance(aer_qasm_sim, shots=10)


expatation_method = qiskit.opflow.AerPauliExpectation()
gradient_method = qiskit.opflow.Gradient()


hf_rx = lambda x: np.array([[np.cos(x/2),-1j*np.sin(x/2)], [-1j*np.sin(x/2),np.cos(x/2)]])
hf_ry = lambda x: np.array([[np.cos(x/2),-np.sin(x/2)], [np.sin(x/2),np.cos(x/2)]])
paulix = np.array([[0,1],[1,0]])
pauliy = np.array([[0,-1j],[1j,0]])
pauliz = np.array([[1,0],[0,-1]])

def hf_numpy_sim(p0, p1, eps=1e-7):
    def hf_circuit(p0_, p1_):
        np0 = np.array([1,1])/np.sqrt(2) #after Hadamard
        np1 = hf_rx(p1_) @ (hf_ry(p0_) @ np0)
        ret = np.dot(np1.conj(), pauliz@np1) + np.dot(np1.conj(), paulix@np1)
        return ret
    ret = hf_circuit(p0, p1)
    grad_p0 = (hf_circuit(p0+eps,p1)-hf_circuit(p0-eps,p1))/(2*eps)
    grad_p1 = (hf_circuit(p0,p1+eps)-hf_circuit(p0,p1-eps))/(2*eps)
    return ret,grad_p0,grad_p1


qiskit_p0 = qiskit.circuit.Parameter("p0") #require_grad=False
qiskit_p1 = qiskit.circuit.Parameter("p1") #require_grad=True
qc1 = qiskit.QuantumCircuit(1)
qc1.h(0)
qc1.ry(qiskit_p0, 0)
qc1.rx(qiskit_p1, 0)

tmp0 = qiskit.opflow.StateFn(qiskit.opflow.PauliSumOp.from_list([("Z", 1.0), ("X", 1.0)]), is_measurement=True)
loss_op0 = tmp0 @ qiskit.opflow.StateFn(qc1)
qnn0 = qiskit_machine_learning.neural_networks.OpflowQNN(loss_op0, [qiskit_p0], [qiskit_p1], expatation_method, gradient_method, qi_sv)
p0_np = np_rng.uniform(size=qnn0.num_inputs) #(np,float64,1)
p1_np = np_rng.uniform(size=qnn0.num_weights)
expectation = qnn0.forward(p0_np, p1_np)
_,grad_p1 = qnn0.backward(p0_np, p1_np)
# batch mode
qnn0.forward([p0_np, p0_np], p1_np)
qnn0.backward([p0_np, p0_np], p1_np)

expectation_,grad_p0_,grad_p1_ = hf_numpy_sim(p0_np[0], p1_np[0])
assert abs(expectation.item()-expectation_)<1e-7
assert abs(grad_p1.item()-grad_p1_)<1e-7

# multiple observables
loss_op1 = qiskit.opflow.ListOp([loss_op0, loss_op0])
op2 = qiskit.opflow.ListOp([loss_op0, loss_op0])
qnn1 = qiskit_machine_learning.neural_networks.OpflowQNN(loss_op1, [qiskit_p0], [qiskit_p1], expatation_method, gradient_method, qi_sv)
qnn1.forward(p0_np, p1_np)
qnn1.backward(p0_np, p1_np)


## feature_map andd ansatz
num_qubit = 3
feature_map = qiskit.circuit.library.ZZFeatureMap(num_qubit, reps=2)
# feature_map.decompose().draw('mpl')
ansatz = qiskit.circuit.library.RealAmplitudes(num_qubit, reps=1)
# ansatz.decompose().draw('mpl')
observable = qiskit.opflow.PauliSumOp.from_list([("Z" * num_qubit, 1)])
qnn2 = qiskit_machine_learning.neural_networks.TwoLayerQNN(num_qubit,
        feature_map=feature_map, ansatz=ansatz, observable=observable, quantum_instance=qi_sv)
p_feature_map = np_rng.uniform(size=qnn2.num_inputs)
p_ansatz = np_rng.uniform(size=qnn2.num_weights)
qnn2.forward(p_feature_map, p_ansatz)
qnn2.backward(p_feature_map, p_ansatz)



## dense qasm_simulator
qc0 = qiskit.circuit.library.RealAmplitudes(num_qubit, entanglement="linear", reps=1)
p_empty = np.empty([0], dtype=np.float64)
# qc0.decompose().draw("mpl")

qnn3 = qiskit_machine_learning.neural_networks.CircuitQNN(qc0, [], qc0.parameters, sparse=True, quantum_instance=qi_qasm)
p_qc0 = np_rng.uniform(size=qnn3.num_weights)
probability = qnn3.forward(p_empty, p_qc0).todense() #based on the qasm(shot=10)
# QNN backward pass, returns a tuple of sparse matrices
qnn3.backward(p_empty, p_qc0) #TODO what's this gradient meaning

parity = lambda x: "{:b}".format(x).count("1") % 2
output_shape = 2  # this is required in case of a callable with dense output
qnn4 = qiskit_machine_learning.neural_networks.CircuitQNN(qc0, [], qc0.parameters,
        sparse=False, interpret=parity, output_shape=output_shape, quantum_instance=qi_qasm)
weights6 = np_rng.uniform(size=qnn4.num_weights)
qnn4.forward(p_empty, weights6)
qnn4.backward(p_empty, weights6)

qnn5 = qiskit_machine_learning.neural_networks.CircuitQNN(qc0, [], qc0.parameters, sampling=True, quantum_instance=qi_qasm)
weights7 = np_rng.uniform(size=qnn5.num_weights)
# results in samples of measured bit strings mapped to integers
qnn5.forward(p_empty, weights7)
qnn5.backward(p_empty, weights7)

qnn6 = qiskit_machine_learning.neural_networks.CircuitQNN(qc0, [], qc0.parameters, sampling=True, interpret=parity, quantum_instance=qi_qasm)
weights8 = np_rng.uniform(size=qnn6.num_weights)
# results in samples of measured bit strings
qnn6.forward(p_empty, weights8)
qnn6.backward(p_empty, weights8)
