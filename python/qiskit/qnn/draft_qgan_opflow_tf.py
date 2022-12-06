# https://learn.qiskit.org/course/machine-learning/quantum-generative-adversarial-networks
import functools
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import tensorflow as tf

import qiskit
import qiskit.opflow
import qiskit.visualization
import qiskit_machine_learning.neural_networks
import qiskit_finance.circuit.library

# plt.ion()

qi_aer_statevector = qiskit.utils.QuantumInstance(qiskit.Aer.get_backend('aer_simulator_statevector'))
opflow_pauli_exp = qiskit.opflow.AerPauliExpectation() # method to calculcate expectation values
opflow_gradient = qiskit.opflow.Gradient()
np_rng = np.random.default_rng()

@functools.lru_cache
def get_all_sorted_key(num_qubit):
    ret = tuple(''.join(x) for x in itertools.product(*(['01']*num_qubit)))
    return ret

def calculate_KL_divergance(p_model, p_target, zero_eps=1e-8):
    all_sorted_key = get_all_sorted_key(len(next(iter(p_target.keys()))))
    np0 = np.array([p_model.get(x,0) for x in all_sorted_key])
    np1 = np.array([p_target.get(x,0) for x in all_sorted_key])
    tmp0 = np.maximum(zero_eps, np1/np.maximum(zero_eps, np0))
    ret = np.sum(np1 * np.log(tmp0))
    return ret

def create_opqnn_circuit():
    num_qubit = 3
    circ_real = qiskit_finance.circuit.library.NormalDistribution(num_qubit, mu=0, sigma=0.15)
    # circ_real = circ_real.decompose().decompose().decompose()

    circ_gen = qiskit.circuit.library.TwoLocal(num_qubit,
            ['ry', 'rz'], 'cz', 'full', reps=2, parameter_prefix='θ_g', name='Generator').decompose()

    theta_d = qiskit.circuit.ParameterVector('θ_d', 12)
    circ_disc = qiskit.QuantumCircuit(num_qubit, name="Discriminator")
    circ_disc.barrier()
    circ_disc.h(0)
    circ_disc.rx(theta_d[0], 0)
    circ_disc.ry(theta_d[1], 0)
    circ_disc.rz(theta_d[2], 0)
    circ_disc.h(1)
    circ_disc.rx(theta_d[3], 1)
    circ_disc.ry(theta_d[4], 1)
    circ_disc.rz(theta_d[5], 1)
    circ_disc.h(2)
    circ_disc.rx(theta_d[6], 2)
    circ_disc.ry(theta_d[7], 2)
    circ_disc.rz(theta_d[8], 2)
    circ_disc.cx(1,2)
    circ_disc.cx(0,2)
    circ_disc.rx(theta_d[9], 2)
    circ_disc.ry(theta_d[10], 2)
    circ_disc.rz(theta_d[11], 2)
    return circ_real, circ_gen, circ_disc


circ_real, circ_gen, circ_disc = create_opqnn_circuit()

num_parameter_disc = circ_disc.num_parameters #theta-d is in front of the theta-gen (alphabetical order)

circ_gen_disc = qiskit.QuantumCircuit(circ_disc.num_qubits)
circ_gen_disc.compose(circ_gen, inplace=True)
circ_gen_disc.compose(circ_disc, inplace=True)

circ_real_disc = qiskit.QuantumCircuit(circ_disc.num_qubits)
circ_real_disc.compose(circ_real, inplace=True)
circ_real_disc.compose(circ_disc, inplace=True)

# construct operator to retrieve Pauli Z expval of the last qubit
H1 = qiskit.opflow.StateFn(qiskit.opflow.PauliSumOp.from_list([('ZII', 1.0)]))
op_gen_disc = ~H1 @ qiskit.opflow.StateFn(circ_gen_disc)
op_real_disc = ~H1  @ qiskit.opflow.StateFn(circ_real_disc)
# 1-2p = <Z>
# p=(1-<Z>)/2

# construct OpflowQNN with the two operators, the input parameters,
# the weight parameters, the expected value, and quantum instance.
# |fake> => |0> => 1 ; |real> => |1> => -1
# gen wants to to minimize this expval
qnn_gen = qiskit_machine_learning.neural_networks.OpflowQNN(op_gen_disc,
            circ_gen_disc.parameters[:num_parameter_disc], circ_gen_disc.parameters[num_parameter_disc:],
            opflow_pauli_exp, opflow_gradient, qi_aer_statevector)
# disc wants to maximize this expval
qnn_fake_disc = qiskit_machine_learning.neural_networks.OpflowQNN(op_gen_disc,
            circ_gen_disc.parameters[num_parameter_disc:], circ_gen_disc.parameters[:num_parameter_disc],
            opflow_pauli_exp, opflow_gradient, qi_aer_statevector)
# disc wants to minimize this expval
qnn_real_disc = qiskit_machine_learning.neural_networks.OpflowQNN(op_real_disc,
            [], circ_gen_disc.parameters[:num_parameter_disc],
            opflow_pauli_exp, opflow_gradient, qi_aer_statevector)

param_gen_tf = tf.Variable(np_rng.uniform(-np.pi, np.pi, size=circ_gen.num_parameters))
param_disc_tf = tf.Variable(np_rng.uniform(-np.pi, np.pi, size=num_parameter_disc))
optimizer_gen = tf.keras.optimizers.Adam(learning_rate=0.02)
optimizer_disc = tf.keras.optimizers.Adam(learning_rate=0.02)

real_prob_dict = qiskit.quantum_info.Statevector(circ_real).probabilities_dict()


num_epoch = 300
num_update_step_disc = 5
history_metric = {'loss_gen':[], 'loss_disc':[], 'kl_div':[]}
best_gen_params = param_gen_tf.numpy().copy()
with tqdm(range(num_epoch)) as pbar:
    for epoch in pbar:
        # Quantum discriminator parameter update
        for ind0 in range(num_update_step_disc):
            grad_fake = qnn_fake_disc.backward(param_gen_tf, param_disc_tf)[1][0,0]
            grad_real = qnn_real_disc.backward([], param_disc_tf)[1][0,0]
            optimizer_disc.apply_gradients([(tf.convert_to_tensor(grad_real - grad_fake), param_disc_tf)])
        prob_fake = (1-qnn_fake_disc.forward(param_gen_tf, param_disc_tf)[0,0])/2
        prob_real = (1-qnn_real_disc.forward([], param_disc_tf)[0,0])/2
        history_metric['loss_disc'].append(prob_fake - prob_real)

        # Quantum generator parameter update
        grad_gcost = qnn_gen.backward(param_disc_tf, param_gen_tf)[1][0,0]
        optimizer_gen.apply_gradients([(tf.convert_to_tensor(grad_gcost), param_gen_tf)])
        prob_fake = (1-qnn_gen.forward(param_disc_tf, param_gen_tf)[0,0])/2
        history_metric['loss_gen'].append(-prob_fake)

        # Track KL and save best performing generator weights
        gen_prob_dict = qiskit.quantum_info.Statevector(circ_gen.bind_parameters(param_gen_tf.numpy())).probabilities_dict()
        history_metric['kl_div'].append(calculate_KL_divergance(gen_prob_dict, real_prob_dict))
        if history_metric['kl_div'][-1]<=min(history_metric['kl_div']):
            best_gen_params = param_gen_tf.numpy().copy()

        pbar.set_postfix({k:f'{v[-1]:.4f}' for k,v in history_metric.items()})


fig, (ax0, ax1) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [0.75, 1]})
fig.suptitle('QGAN training stats')
fig.supxlabel('Training step')
ax0.plot(history_metric['loss_gen'], label="Generator loss")
ax0.plot(history_metric['loss_disc'], label="Discriminator loss", color="C3")
ax0.legend()
ax0.set(ylabel='Loss')
ax1.plot(history_metric['kl_div'], label="KL Divergence (zero is best)", color="C1")
ax1.set(ylabel='KL Divergence')
ax1.legend()
fig.tight_layout()


gen_prob_dict = qiskit.quantum_info.Statevector(circ_gen.bind_parameters(best_gen_params)).probabilities_dict()
fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True, figsize=(9,3))
qiskit.visualization.plot_histogram(gen_prob_dict, ax=ax0)
ax0.set_title("Trained generator distribution")
qiskit.visualization.plot_histogram(real_prob_dict, ax=ax1)
ax1.set_title("Real distribution")
ax1.set_ylim([0,.5])
fig.tight_layout()
