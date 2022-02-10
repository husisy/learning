import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import qiskit
import qiskit.algorithms
import qiskit.providers.aer
import qiskit.ignis.mitigation
import qiskit.test.mock

def vqe_callback_hook(cout_list, mean_list):
    def hf0(eval_count, parameters, mean, std):
        cout_list.append(eval_count)
        mean_list.append(mean)
    return hf0


# https://qiskit.org/documentation/tutorials/algorithms/01_algorithms_introduction.html
H2_op = ((-1.052373245772859 * qiskit.opflow.I ^ qiskit.opflow.I)
        + (0.39793742484318045 * qiskit.opflow.I ^ qiskit.opflow.Z)
        + (-0.39793742484318045 * qiskit.opflow.Z ^ qiskit.opflow.I)
        + (-0.01128010425623538 * qiskit.opflow.Z ^ qiskit.opflow.Z)
        + (0.18093119978423156 * qiskit.opflow.X ^ qiskit.opflow.X))
# qiskit.utils.algorithm_globals.random_seed = 50
backend = qiskit.Aer.get_backend('aer_simulator_statevector') #aer_simulator statevector_simulator aer_simulator_statevector
qi = qiskit.utils.QuantumInstance(backend) #seed_transpiler=50, seed_simulator=50
ansatz = qiskit.circuit.library.TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
# ansatz.draw()
slsqp = qiskit.algorithms.optimizers.SLSQP(maxiter=1000)
vqe = qiskit.algorithms.VQE(ansatz, optimizer=slsqp, quantum_instance=qi)
# vqe.quantum_instance = qi
result = vqe.compute_minimum_eigenvalue(H2_op)

npme = qiskit.algorithms.NumPyMinimumEigensolver()
ref_value = npme.compute_minimum_eigenvalue(operator=H2_op).eigenvalue.real
assert abs(ref_value-result.optimal_value) < 1e-3


# https://qiskit.org/documentation/tutorials/algorithms/02_vqe_convergence.html
H2_op = ((-1.052373245772859 * qiskit.opflow.I ^ qiskit.opflow.I)
        + (0.39793742484318045 * qiskit.opflow.I ^ qiskit.opflow.Z)
        + (-0.39793742484318045 * qiskit.opflow.Z ^ qiskit.opflow.I)
        + (-0.01128010425623538 * qiskit.opflow.Z ^ qiskit.opflow.Z)
        + (0.18093119978423156 * qiskit.opflow.X ^ qiskit.opflow.X))
tmp0 = ['COBYLA', 'L_BFGS_B', 'SLSQP']
optimizer_dict = {x:getattr(qiskit.algorithms.optimizers,x)(maxiter=80) for x in tmp0}
converge_cnts = dict()
converge_vals = dict()
for key,optimizer in optimizer_dict.items():
    ansatz = qiskit.circuit.library.TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
    counts = []
    values = []
    qi = qiskit.utils.QuantumInstance(backend=qiskit.Aer.get_backend('statevector_simulator'))
    vqe = qiskit.algorithms.VQE(ansatz, optimizer, callback=vqe_callback_hook(counts,values), quantum_instance=qi)
    vqe.compute_minimum_eigenvalue(operator=H2_op)
    converge_cnts[key] = np.asarray(counts)
    converge_vals[key] = np.asarray(values)

fig,ax = plt.subplots()
for key in optimizer_dict.keys():
    ax.plot(converge_cnts[key], converge_vals[key], label=key)
ax.set_ylabel('Energy')
ax.set_title('Energy convergence for various optimizers')
ax.legend()

ansatz = qiskit.circuit.library.TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
optimizer = qiskit.algorithms.optimizers.SLSQP(maxiter=60)
counts = []
values = []
qi = qiskit.utils.QuantumInstance(backend=qiskit.Aer.get_backend('statevector_simulator'))
vqe = qiskit.algorithms.VQE(ansatz, optimizer, callback=vqe_callback_hook(counts,values),
        gradient=qiskit.opflow.gradients.Gradient(grad_method='fin_diff'), quantum_instance=qi)
result = vqe.compute_minimum_eigenvalue(operator=H2_op)
result.eigenvalue.real #-1.07998


# https://qiskit.org/documentation/tutorials/algorithms/03_vqe_simulation_with_noise.html
H2_op = ((-1.052373245772859 * qiskit.opflow.I ^ qiskit.opflow.I)
        + (0.39793742484318045 * qiskit.opflow.I ^ qiskit.opflow.Z)
        + (-0.39793742484318045 * qiskit.opflow.Z ^ qiskit.opflow.I)
        + (-0.01128010425623538 * qiskit.opflow.Z ^ qiskit.opflow.Z)
        + (0.18093119978423156 * qiskit.opflow.X ^ qiskit.opflow.X))

# without noise
backend = qiskit.Aer.get_backend('aer_simulator')
qi = qiskit.utils.QuantumInstance(backend=backend)
counts = []
values = []
ansatz = qiskit.circuit.library.TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
spsa = qiskit.algorithms.optimizers.SPSA(maxiter=125)
vqe = qiskit.algorithms.VQE(ansatz, optimizer=spsa, callback=vqe_callback_hook(counts,values), quantum_instance=qi)
result = vqe.compute_minimum_eigenvalue(operator=H2_op)
result.eigenvalue.real

fig,ax = plt.subplots()
ax.plot(counts, values)
ax.set_ylabel('energy')
ax.set_title('without noise')

# noise
backend = qiskit.Aer.get_backend('aer_simulator')
counts1 = []
values1 = []
device = qiskit.providers.aer.QasmSimulator.from_backend(qiskit.test.mock.FakeVigo())
coupling_map = device.configuration().coupling_map
noise_model = qiskit.providers.aer.noise.NoiseModel.from_backend(device)
qi = qiskit.utils.QuantumInstance(backend=backend, coupling_map=coupling_map, noise_model=noise_model,)
ansatz = qiskit.circuit.library.TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
spsa = qiskit.algorithms.optimizers.SPSA(maxiter=125)
vqe = qiskit.algorithms.VQE(ansatz, optimizer=spsa, callback=vqe_callback_hook(counts1,values1), quantum_instance=qi)
result1 = vqe.compute_minimum_eigenvalue(operator=H2_op)
result1.eigenvalue.real

fig,ax = plt.subplots()
ax.plot(counts1, values1)
ax.set_ylabel('energy')
ax.set_title('with noise')

# noise, error mitigation
counts2 = []
values2 = []
qi = qiskit.utils.QuantumInstance(backend=backend, coupling_map=coupling_map, noise_model=noise_model,
        measurement_error_mitigation_cls=qiskit.ignis.mitigation.measurement.CompleteMeasFitter,
        cals_matrix_refresh_period=30)
ansatz = qiskit.circuit.library.TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
spsa = qiskit.algorithms.optimizers.SPSA(maxiter=125)
vqe = qiskit.algorithms.VQE(ansatz, optimizer=spsa, callback=vqe_callback_hook(counts2,values2), quantum_instance=qi)
result2 = vqe.compute_minimum_eigenvalue(operator=H2_op)
result2.eigenvalue.real

fig,ax = plt.subplots()
ax.plot(counts2, values2)
ax.set_ylabel('energy')
ax.set_title('with noise, measurement error mitigation enabled')


## advanced VQE usage https://qiskit.org/documentation/tutorials/algorithms/04_vqe_advanced.html
H2_op = ((-1.052373245772859 * qiskit.opflow.I ^ qiskit.opflow.I)
        + (0.39793742484318045 * qiskit.opflow.I ^ qiskit.opflow.Z)
        + (-0.39793742484318045 * qiskit.opflow.Z ^ qiskit.opflow.I)
        + (-0.01128010425623538 * qiskit.opflow.Z ^ qiskit.opflow.Z)
        + (0.18093119978423156 * qiskit.opflow.X ^ qiskit.opflow.X))

qi = qiskit.utils.QuantumInstance(qiskit.Aer.get_backend('statevector_simulator'))
ansatz = qiskit.circuit.library.TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
slsqp = qiskit.algorithms.optimizers.SLSQP(maxiter=1000)
vqe = qiskit.algorithms.VQE(ansatz, optimizer=slsqp, quantum_instance=qi)
result = vqe.compute_minimum_eigenvalue(operator=H2_op)
initial_pt = result.optimal_point
result.optimal_value

# set initial_point
qi = qiskit.utils.QuantumInstance(qiskit.Aer.get_backend('statevector_simulator'))
ansatz = qiskit.circuit.library.TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
slsqp = qiskit.algorithms.optimizers.SLSQP(maxiter=1000)
vqe = qiskit.algorithms.VQE(ansatz, optimizer=slsqp, initial_point=initial_pt, quantum_instance=qi)
result1 = vqe.compute_minimum_eigenvalue(operator=H2_op)

# include_custom=True
qi = qiskit.utils.QuantumInstance(qiskit.Aer.get_backend('aer_simulator'))
ansatz = qiskit.circuit.library.TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
slsqp = qiskit.algorithms.optimizers.SLSQP(maxiter=1000)
vqe = qiskit.algorithms.VQE(ansatz, optimizer=slsqp, quantum_instance=qi, include_custom=True)
result = vqe.compute_minimum_eigenvalue(operator=H2_op)
optimal_value1 = result.optimal_value

# the shot noise confusing the SLSQP optimizer
qi = qiskit.utils.QuantumInstance(qiskit.Aer.get_backend('aer_simulator'))
ansatz = qiskit.circuit.library.TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
slsqp = qiskit.algorithms.optimizers.SLSQP(maxiter=1000)
vqe = qiskit.algorithms.VQE(ansatz, optimizer=slsqp, quantum_instance=qi)
result = vqe.compute_minimum_eigenvalue(operator=H2_op)
optimal_value = result.optimal_value #may larger then -1.857

qi = qiskit.utils.QuantumInstance(qiskit.Aer.get_backend('aer_simulator'))
ansatz = qiskit.circuit.library.TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
slsqp = qiskit.algorithms.optimizers.SPSA(maxiter=100) #designed to work in noisy environments
vqe = qiskit.algorithms.VQE(ansatz, optimizer=slsqp, quantum_instance=qi)
result = vqe.compute_minimum_eigenvalue(operator=H2_op)

# expectation=
qi = qiskit.utils.QuantumInstance(qiskit.Aer.get_backend('aer_simulator'))
ansatz = qiskit.circuit.library.TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
slsqp = qiskit.algorithms.optimizers.SLSQP(maxiter=1000)
vqe = qiskit.algorithms.VQE(ansatz, optimizer=slsqp, quantum_instance=qi, expectation=qiskit.opflow.AerPauliExpectation())
result = vqe.compute_minimum_eigenvalue(operator=H2_op)

qi = qiskit.utils.QuantumInstance(qiskit.Aer.get_backend('aer_simulator'))
ansatz = qiskit.circuit.library.TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
slsqp = qiskit.algorithms.optimizers.SPSA(maxiter=100)
vqe = qiskit.algorithms.VQE(ansatz, optimizer=slsqp, quantum_instance=qi, expectation=qiskit.opflow.PauliExpectation(group_paulis=False))
result = vqe.compute_minimum_eigenvalue(operator=H2_op)

# matrix product state
qi = qiskit.utils.QuantumInstance(qiskit.providers.aer.QasmSimulator(method='matrix_product_state'), shots=1)
ansatz = qiskit.circuit.library.TwoLocal(rotation_blocks='ry', entanglement_blocks='cz')
slsqp = qiskit.algorithms.optimizers.SLSQP(maxiter=1000)
vqe = qiskit.algorithms.VQE(ansatz, optimizer=slsqp, quantum_instance=qi, include_custom=True)
result = vqe.compute_minimum_eigenvalue(operator=H2_op)
