# https://learn.qiskit.org/course/machine-learning/introduction
import numpy as np
import matplotlib.pyplot as plt
import qiskit
import qiskit.opflow
import qiskit.algorithms
# import qiskit.visualization
import qiskit.visualization.bloch
# from qiskit import QuantumCircuit, assemble, Aer
# from qiskit.visualization import plot_histogram

np_rng = np.random.default_rng()
aer_simulator = qiskit.Aer.get_backend('qasm_simulator')
aer_quantum_instance = qiskit.utils.QuantumInstance(aer_simulator, shots=8192)
sampler = qiskit.opflow.CircuitSampler(aer_quantum_instance)
opflow_pauli = qiskit.opflow.PauliExpectation()
sampler_cache_all = qiskit.opflow.CircuitSampler(aer_quantum_instance, caching="all")

theta = qiskit.circuit.Parameter('θ')
circ0 = qiskit.circuit.QuantumCircuit(2)
circ0.rz(theta, 0)
circ0.crz(theta, 0, 1)
# circ0.draw()


theta = qiskit.circuit.ParameterVector('θ', length=2)
circ0 = qiskit.circuit.QuantumCircuit(2)
circ0.rz(theta[0], 0)
circ0.crz(theta[1], 0, 1)



def state_to_bloch(np0):
    # Converts state vectors to points on the Bloch sphere
    phi = np.angle(np0[1])-np.angle(np0[0])
    theta = 2*np.arccos(np.abs(np0[0]))
    ret = np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)
    return ret

# https://learn.qiskit.org/course/machine-learning/parameterized-quantum-circuits
circ_param = qiskit.circuit.ParameterVector('θ', length=2)
circ_param_value = np_rng.uniform(0, 2*np.pi, size=(1000,2))

circ0 = qiskit.circuit.QuantumCircuit(1)
circ0.h(0)
circ0.rz(circ_param[0], 0)
circ1 = qiskit.circuit.QuantumCircuit(1)
circ1.h(0)
circ1.rz(circ_param[0], 0)
circ1.rx(circ_param[1], 0)
circ_list = [circ0, circ1]

fig = plt.figure(figsize=plt.figaspect(1/2)) #(width,height)
ax_list = []
hbloch_list = []
for ind0 in range(2):
    ax = fig.add_subplot(1, 2, ind0+1, projection='3d')
    hbloch = qiskit.visualization.bloch.Bloch(axes=ax)
    hbloch.point_color = ['tab:blue']
    hbloch.point_marker = ['o']
    hbloch.point_size = [2]
    for ind1 in range(len(circ_param_value)):
        tmp0 = {x:y for x,y in zip(circ_param[:(ind0+1)], circ_param_value[ind1])}
        tmp1 = qiskit.quantum_info.Statevector.from_instruction(circ_list[ind0].bind_parameters(tmp0))
        hbloch.add_points(state_to_bloch(tmp1.data))
    hbloch.show()
    ax_list.append(ax)
    hbloch_list.append(hbloch)



# https://learn.qiskit.org/course/machine-learning/parameterized-quantum-circuits
qc_zz = qiskit.circuit.library.ZZFeatureMap(3, reps=1, insert_barriers=True)
# qc_zz.decompose().draw()

qc_twolocal = qiskit.circuit.library.TwoLocal(num_qubits=3, reps=2, rotation_blocks=['ry','rz'],
                entanglement_blocks='cz', skip_final_rotation_layer=True,
                insert_barriers=True)
qc_13 = qiskit.circuit.library.TwoLocal(3, rotation_blocks='ry',
                 entanglement_blocks='crz', entanglement='sca',
                 reps=3, skip_final_rotation_layer=True,
                 insert_barriers=True)

# rotation block:
rot = qiskit.circuit.QuantumCircuit(2)
params = qiskit.circuit.ParameterVector('r', 2)
rot.ry(params[0], 0)
rot.rz(params[1], 1)

# entanglement block:
ent = qiskit.circuit.QuantumCircuit(4)
params = qiskit.circuit.ParameterVector('e', 3)
ent.crx(params[0], 0, 1)
ent.crx(params[1], 1, 2)
ent.crx(params[2], 2, 3)

qc_nlocal = qiskit.circuit.library.NLocal(num_qubits=6, rotation_blocks=rot,
                   entanglement_blocks=ent, entanglement='linear',
                   skip_final_rotation_layer=True, insert_barriers=True)


# basic encoding
desired_state = [0, 0, 0, 0, 0, 1/np.sqrt(2), 0, 1/np.sqrt(2)]
circ = qiskit.circuit.QuantumCircuit(3)
circ.initialize(desired_state, [0,1,2])
# circ.decompose().decompose().decompose().decompose().decompose().draw()

# amplitude encoding
tmp0 = np.array([1.5, 0, -2, 3])
desired_state = tmp0 / np.linalg.rnom(tmp0)
circ0 = qiskit.circuit.QuantumCircuit(2)
circ0.initialize(desired_state, [0,1])
# circ0.decompose().decompose().decompose().decompose().decompose().draw()

# angle encoding
circ0 = qiskit.circuit.QuantumCircuit(3)
circ0.ry(0, 0)
circ0.ry(2*np.pi/4, 1)
circ0.ry(2*np.pi/2, 2)

# arbitrary encoding
circ0 = qiskit.circuit.library.EfficientSU2(num_qubits=3, reps=1, insert_barriers=True)
x = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
circ1 = circ0.bind_parameters(x)
# circ0.decompose().draw()
# circ1.decompose().draw()


## gradient
ansatz = qiskit.circuit.library.RealAmplitudes(num_qubits=2, reps=1, entanglement='linear').decompose() #wrong result if not decompose
# ansatz.draw()
hamiltonian = qiskit.opflow.Z ^ qiskit.opflow.Z
expectation = qiskit.opflow.StateFn(hamiltonian, is_measurement=True) @ qiskit.opflow.StateFn(ansatz)
pauli_basis = opflow_pauli.convert(expectation)

point = np_rng.uniform(size=ansatz.num_parameters)
INDEX = 2
zero_eps = 0.1

e_i = np.identity(point.size)[:, INDEX] #identity vector with a 1 at index ``INDEX``, otherwise 0
plus = point + zero_eps * e_i
minus = point - zero_eps * e_i
hf0 = lambda theta: sampler.convert(pauli_basis, params=dict(zip(ansatz.parameters, theta))).eval().real
grad_fd = (hf0(plus) - hf0(minus)) / (2 * zero_eps)

plus = point + e_i*(np.pi/2)
minus = point - e_i*(np.pi/2)
grad_shift_rule = (hf0(plus) - hf0(minus)) / 2

shifter = qiskit.opflow.Gradient('fin_diff', analytic=False, epsilon=zero_eps) #fin_diff param_shift(default)
grad = shifter.convert(expectation, params=ansatz.parameters[INDEX])
grad_opflow = sampler.convert(grad, dict(zip(ansatz.parameters, point))).eval().real
print(grad_fd, grad_shift_rule, grad_opflow)

tmp0 = opflow_pauli.convert(qiskit.opflow.Gradient().convert(expectation)) #fin_diff param_shift(default)
grad_opflow = np.array(sampler.convert(tmp0, dict(zip(ansatz.parameters, point))).eval()).real




## compare gradient
ansatz = qiskit.circuit.library.RealAmplitudes(num_qubits=2, reps=1, entanglement='linear').decompose() #wrong result if not decompose
hamiltonian = qiskit.opflow.Z ^ qiskit.opflow.Z
expectation = qiskit.opflow.StateFn(hamiltonian, is_measurement=True) @ qiskit.opflow.StateFn(ansatz)
pauli_basis = opflow_pauli.convert(expectation)
gradient_in_pauli_basis = opflow_pauli.convert(qiskit.opflow.Gradient().convert(expectation))
hf_fval = lambda theta: sampler.convert(pauli_basis, params=dict(zip(ansatz.parameters, theta))).eval().real
hf_gradient = lambda theta: np.array(sampler.convert(gradient_in_pauli_basis, params=dict(zip(ansatz.parameters, theta))).eval()).real

natural_gradient = qiskit.opflow.NaturalGradient(regularization='ridge').convert(expectation)
natural_gradient_in_pauli_basis = opflow_pauli.convert(natural_gradient)
hf_gradient_natural = lambda theta: np.array(sampler.convert(natural_gradient, params=dict(zip(ansatz.parameters, theta))).eval()).real

# initial_point = np.random.random(ansatz.num_parameters)
initial_point = np.array([0.43253681, 0.09507794, 0.42805949, 0.34210341])

class OptimizerLog:
    """Log to store optimizer's intermediate results"""
    def __init__(self):
        self.loss = []
    def update(self, _nfevs, _theta, ftheta, grad_norm, *args): #spsa pass extra args here
        """Save intermediate results. Optimizers pass many values but we only store the third ."""
        self.loss.append(ftheta)
gd_log = OptimizerLog()
gd = qiskit.algorithms.optimizers.GradientDescent(maxiter=300, learning_rate=0.01, callback=gd_log.update)
result = gd.minimize(fun=hf_fval, x0=initial_point, jac=hf_gradient)

qng_log = OptimizerLog()
qng = qiskit.algorithms.optimizers.GradientDescent(maxiter=300, learning_rate=0.01, callback=qng_log.update)
result = qng.minimize(hf_fval, initial_point, hf_gradient_natural) #slow

spsa_log = OptimizerLog()
spsa = qiskit.algorithms.optimizers.SPSA(maxiter=300, learning_rate=0.01, perturbation=0.01, callback=spsa_log.update)
result = spsa.minimize(hf_fval, initial_point)

qnspsa_log = OptimizerLog()
fidelity = qiskit.algorithms.optimizers.QNSPSA.get_fidelity(ansatz, aer_quantum_instance, expectation=opflow_pauli)
qnspsa = qiskit.algorithms.optimizers.QNSPSA(fidelity, maxiter=300, learning_rate=0.01, perturbation=0.01, callback=qnspsa_log.update)
result = qnspsa.minimize(hf_fval, initial_point)

# automatically calibrate the learning rate
autospsa_log = OptimizerLog()
autospsa = qiskit.algorithms.optimizers.SPSA(maxiter=300, learning_rate=None, perturbation=None, callback=autospsa_log.update)
result = autospsa.minimize(hf_fval, initial_point)

fig,ax = plt.subplots(figsize=(7, 3))
ax.plot(gd_log.loss, 'C0', label='vanilla gradient descent')
ax.plot(qng_log.loss, 'C1', label='quantum natural gradient')
plt.plot(spsa_log.loss, 'C0', ls='--', label='SPSA')
plt.plot(qnspsa_log.loss, 'C1', ls='--', label='QN-SPSA')
plt.plot(autospsa_log.loss, 'C3', label='Power-law SPSA')
ax.axhline(-1, c='C3', ls='--', label='target')
ax.set_ylabel('loss')
ax.set_xlabel('iterations')
ax.legend()


## barren plateaus
def sample_gradients(num_qubits, reps, local=False):
    """Sample the gradient of our model for ``num_qubits`` qubits and ``reps`` repetitions.
    We sample 100 times for random parameters and compute the gradient of the first RY rotation gate.
    """
    index = num_qubits - 1
    if local:
        operator = qiskit.opflow.Z ^ qiskit.opflow.Z ^ (qiskit.opflow.I ^ (num_qubits - 2))
    else: #global
        operator = qiskit.opflow.Z ^ num_qubits
    ansatz = qiskit.circuit.library.RealAmplitudes(num_qubits, entanglement='linear', reps=reps)
    expectation = qiskit.opflow.StateFn(operator, is_measurement=True).compose(qiskit.opflow.StateFn(ansatz))
    grad = qiskit.opflow.Gradient().convert(expectation, params=ansatz.parameters[index])
    tmp0 = np.random.uniform(0, np.pi, size=(100,ansatz.num_parameters))
    ret = [sampler.convert(grad, dict(zip(ansatz.parameters, x))).eval() for x in tmp0]
    return ret

num_qubits = list(range(2, 13))
linear_depth_global_gradients = np.array([sample_gradients(n, n) for n in num_qubits])
fixed_depth_global_gradients = np.array([sample_gradients(n, 1) for n in num_qubits])
linear_depth_local_gradients = np.array([sample_gradients(n, n, local=True) for n in num_qubits])
fixed_depth_local_gradients = np.array([sample_gradients(n, 1, local=True) for n in num_qubits])

fit = np.polyfit(num_qubits, np.log(np.var(fixed_depth_local_gradients, axis=1)), deg=1)

x = np.linspace(num_qubits[0], num_qubits[-1], 200)
fig,ax = plt.subplots(figsize=(7, 3))
ax.plot(num_qubits, np.var(linear_depth_global_gradients, axis=1), 'o-', label='global cost, linear depth')
ax.plot(num_qubits, np.var(fixed_depth_global_gradients, axis=1), 'o-', label='global cost, constant depth')
ax.plot(num_qubits, np.var(linear_depth_local_gradients, axis=1), 'o-', label='local cost, linear depth')
ax.plot(num_qubits, np.var(fixed_depth_local_gradients, axis=1), 'o-', label='local cost, constant depth')
ax.plot(x, np.exp(fit[0] * x + fit[1]), '--', c='C3', label=f'exponential fit w/ {fit[0]:.2f}')
ax.set_xlabel('number of qubits')
ax.set_yscale('log')
ax.set_ylabel(r'$\mathrm{Var}[\partial_{\theta 1}\langle E(\theta)\rangle]$')
ax.legend()


## layerwise training
NUM_QUBITS = 6
OPERATOR = qiskit.opflow.Z ^ qiskit.opflow.Z ^ (qiskit.opflow.I ^ (NUM_QUBITS - 4))

def minimize(circuit, optimizer):
    """
    Args:
        circuit (QuantumCircuit): (Partially bound) ansatz circuit to train
        optimizer (Optimizer): Algorithm to use to minimize exp. value
    Returns:
        OptimizerResult: Result of minimization
    """
    initial_point = np.random.random(circuit.num_parameters)
    exp = opflow_pauli.convert(qiskit.opflow.StateFn(OPERATOR, is_measurement=True) @ qiskit.opflow.StateFn(circuit))
    grad = opflow_pauli.convert(qiskit.opflow.Gradient().convert(exp))
    hf_fval = lambda theta: np.real(sampler_cache_all.convert(exp, dict(zip(circuit.parameters, theta))).eval())
    hf_gradient = lambda theta: np.real(sampler_cache_all.convert(grad, dict(zip(circuit.parameters, theta))).eval())
    ret = optimizer.minimize(hf_fval, initial_point, hf_gradient)
    return ret

ansatz = qiskit.circuit.library.RealAmplitudes(4, entanglement='linear')
optimizer = qiskit.algorithms.optimizers.GradientDescent(maxiter=50)
max_num_layers = 4
optimal_parameters = []
for reps in range(max_num_layers):
    ansatz.reps = reps
    #fix the already optimized parameters
    partially_bound = ansatz.bind_parameters(dict(zip(ansatz.parameters, optimal_parameters)))
    result = minimize(partially_bound, optimizer)
    optimal_parameters += list(result.x)
    print('Layer:', reps, ' Best Value:', result.fun)
