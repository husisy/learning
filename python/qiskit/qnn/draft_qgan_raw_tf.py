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

# plt.ion()

qi_aer_statevector = qiskit.utils.QuantumInstance(qiskit.Aer.get_backend('aer_simulator_statevector'))
opflow_pauli_exp = qiskit.opflow.AerPauliExpectation() # method to calculcate expectation values
opflow_gradient = qiskit.opflow.Gradient()
np_rng = np.random.default_rng()

def create_qgan_circuit(num_qubit_g=2):
    assert num_qubit_g==2

    circ_real = qiskit.QuantumCircuit(num_qubit_g)
    circ_real.h(0)
    circ_real.cx(0, 1)

    # decompose into standard gates
    circ_gen = qiskit.circuit.library.TwoLocal(num_qubit_g, ['ry','rz'],
                'cz', 'full', reps=2, parameter_prefix='θ_g', name='Generator').decompose()

    theta_d = qiskit.circuit.ParameterVector('θ_d', 12)
    circ_disc = qiskit.QuantumCircuit(num_qubit_g+1, name="Discriminator")
    circ_disc.barrier()
    circ_disc.h(0)
    circ_disc.rx(theta_d[0], 0)
    circ_disc.ry(theta_d[1], 0)
    circ_disc.rz(theta_d[2], 0)
    circ_disc.rx(theta_d[3], 1)
    circ_disc.ry(theta_d[4], 1)
    circ_disc.rz(theta_d[5], 1)
    circ_disc.rx(theta_d[6], 2)
    circ_disc.ry(theta_d[7], 2)
    circ_disc.rz(theta_d[8], 2)
    circ_disc.cx(0, 2)
    circ_disc.cx(1, 2)
    circ_disc.rx(theta_d[9], 2)
    circ_disc.ry(theta_d[10], 2)
    circ_disc.rz(theta_d[11], 2)

    circ_gen_disc = qiskit.QuantumCircuit(num_qubit_g+1)
    circ_gen_disc.compose(circ_gen, inplace=True)
    circ_gen_disc.compose(circ_disc, inplace=True)

    circ_real_disc = qiskit.QuantumCircuit(num_qubit_g+1)
    circ_real_disc.compose(circ_real, inplace=True)
    circ_real_disc.compose(circ_disc, inplace=True)
    return circ_real, circ_gen, circ_disc, circ_gen_disc, circ_real_disc


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


hf_sum_q1 = lambda x: x[4:].sum(axis=0)

circ_real, circ_gen, circ_disc, circ_gen_disc, circ_real_disc = create_qgan_circuit()
num_parameter_disc = circ_disc.num_parameters #theta-d is in front of the theta-gen (alphabetical order)

# frozen theta-d, trainable theta-g
qnn_gen = qiskit_machine_learning.neural_networks.CircuitQNN(circ_gen_disc,
            circ_gen_disc.parameters[:num_parameter_disc], circ_gen_disc.parameters[num_parameter_disc:],
            quantum_instance=qi_aer_statevector)

# frozen theta-g, trainable theta-d
qnn_fake_disc = qiskit_machine_learning.neural_networks.CircuitQNN(circ_gen_disc,
            circ_gen_disc.parameters[num_parameter_disc:], circ_gen_disc.parameters[:num_parameter_disc],
            quantum_instance=qi_aer_statevector)

# trainable theta-d
qnn_real_disc = qiskit_machine_learning.neural_networks.CircuitQNN(circ_real_disc,
            [], circ_gen_disc.parameters[:num_parameter_disc],
            quantum_instance=qi_aer_statevector)


# Initialize parameters
param_disc_tf = tf.Variable(np_rng.uniform(-np.pi, np.pi, size=num_parameter_disc))
param_gen_tf = tf.Variable(np_rng.uniform(-np.pi, np.pi, size=circ_gen.num_parameters))

real_prob_dict = qiskit.quantum_info.Statevector(circ_real).probabilities_dict()
print('real:', real_prob_dict)
tmp0 = qiskit.quantum_info.Statevector(circ_gen.bind_parameters(param_gen_tf.numpy())).probabilities_dict()
print('initial generator:', tmp0)
# qiskit.visualization.plot_histogram(tmp0, ax=ax0)

optimizer_gen = tf.keras.optimizers.Adam(learning_rate=0.02)
optimizer_disc = tf.keras.optimizers.Adam(learning_rate=0.02)

num_epoch = 100
num_update_step_disc = 5
history_metric = {'loss_gen':[], 'loss_disc':[], 'kl_div':[]}
best_gen_params = param_gen_tf.numpy().copy()
with tqdm(range(num_epoch)) as pbar:
    for ind0 in pbar:
        # train discriminator
        for _ in range(num_update_step_disc):
            # Earth move distance (Wasserstein distance), WGAN https://en.wikipedia.org/wiki/Earth_mover%27s_distance
            d_fake = hf_sum_q1(qnn_fake_disc.backward(param_gen_tf, param_disc_tf)[1][0])
            d_real = hf_sum_q1(qnn_real_disc.backward([], param_disc_tf)[1][0])
            optimizer_disc.apply_gradients([(tf.convert_to_tensor(d_fake-d_real), param_disc_tf)])
        tmp0 = np.concatenate([param_disc_tf.numpy(), param_gen_tf.numpy()])
        prob_fake = hf_sum_q1(qiskit.quantum_info.Statevector(circ_gen_disc.bind_parameters(tmp0)).probabilities())
        prob_real = hf_sum_q1(qiskit.quantum_info.Statevector(circ_real_disc.bind_parameters(param_disc_tf.numpy())).probabilities())
        history_metric['loss_disc'].append(prob_fake - prob_real)

        # train generator
        tmp0 = -hf_sum_q1(qnn_gen.backward(param_disc_tf, param_gen_tf)[1][0])
        optimizer_gen.apply_gradients([(tf.convert_to_tensor(tmp0), param_gen_tf)])
        tmp0 = np.concatenate([param_disc_tf.numpy(), param_gen_tf.numpy()])
        prob_fake = hf_sum_q1(qiskit.quantum_info.Statevector(circ_gen_disc.bind_parameters(tmp0)).probabilities())
        history_metric['loss_gen'].append(-prob_fake)

        # track KL divergence and save best performing generator weights
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

print('real:', qiskit.quantum_info.Statevector(circ_real).probabilities_dict())
tmp0 = qiskit.quantum_info.Statevector(circ_gen.bind_parameters(best_gen_params)).probabilities_dict()
print('final generator:', tmp0)
# qiskit.visualization.plot_histogram(tmp0, ax=ax0)

