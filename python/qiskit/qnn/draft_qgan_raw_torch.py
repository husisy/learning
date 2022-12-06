# https://learn.qiskit.org/course/machine-learning/quantum-generative-adversarial-networks
import functools
import itertools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# import tensorflow as tf
import torch

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
    return circ_real, circ_gen, circ_disc


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


class QGAN(torch.nn.Module):
    def __init__(self, circ_real, circ_gen, circ_disc):
        super().__init__()
        self.circ_real = circ_real
        self.circ_gen = circ_gen
        self.circ_disc = circ_disc
        self.num_qubit = circ_disc.num_qubits

        # make circuit
        circ_gen_disc = qiskit.QuantumCircuit(circ_disc.num_qubits)
        circ_gen_disc.compose(circ_gen, inplace=True)
        circ_gen_disc.compose(circ_disc, inplace=True)
        circ_real_disc = qiskit.QuantumCircuit(circ_disc.num_qubits)
        circ_real_disc.compose(circ_real, inplace=True)
        circ_real_disc.compose(circ_disc, inplace=True)
        self.circ_gen_disc = circ_gen_disc
        self.circ_real_disc = circ_real_disc

        # construct operator to retrieve Pauli Z expval of the last qubit
        # 1-2p = <Z>,  p=(1-<Z>)/2
        H1 = qiskit.opflow.StateFn(qiskit.opflow.PauliSumOp.from_list([('ZII', 1.0)]))
        self.op_gen_disc = ~H1 @ qiskit.opflow.StateFn(self.circ_gen_disc)
        self.op_real_disc = ~H1  @ qiskit.opflow.StateFn(self.circ_real_disc)

        # construct qnn model
        # theta-d is in front of the theta-gen (alphabetical order)
        # |fake> => |0> => 1 ; |real> => |1> => -1
        theta_d = circ_gen_disc.parameters[:circ_disc.num_parameters]
        theta_g = circ_gen_disc.parameters[circ_disc.num_parameters:]
        # frozen theta-d, trainable theta-g
        self.qnn_gen = qiskit_machine_learning.neural_networks.CircuitQNN(circ_gen_disc,
                    theta_d, theta_g, quantum_instance=qi_aer_statevector)
        # frozen theta-g, trainable theta-d
        self.qnn_fake_disc = qiskit_machine_learning.neural_networks.CircuitQNN(circ_gen_disc,
                    theta_g, theta_d, quantum_instance=qi_aer_statevector)
        # trainable theta-d
        self.qnn_real_disc = qiskit_machine_learning.neural_networks.CircuitQNN(circ_real_disc,
                    [], theta_d, quantum_instance=qi_aer_statevector)

        # trainable parameter
        np_rng = np.random.default_rng()
        hf0 = lambda *size: torch.nn.Parameter(torch.tensor(np_rng.uniform(-np.pi, np.pi, size=size), dtype=torch.float64))
        self.theta_gen = hf0(circ_gen.num_parameters)
        self.theta_disc = hf0(circ_disc.num_parameters)
        self.theta_gen.grad = torch.zeros_like(self.theta_gen)
        self.theta_disc.grad = torch.zeros_like(self.theta_disc)

        # equivalent to measure 1 on last qubit
        tmp0 = 2**(circ_disc.num_qubits-1)
        self._sum_q1 = lambda x,N=tmp0: x[N:].sum(axis=0)

    def get_theta_np(self):
        theta_gen_np = self.theta_gen.detach().numpy()
        theta_disc_np = self.theta_disc.detach().numpy()
        return theta_gen_np,theta_disc_np

    def backward_disc(self):
        theta_gen_np,theta_disc_np = self.get_theta_np()
        grad_fake = self._sum_q1(self.qnn_fake_disc.backward(theta_gen_np, theta_disc_np)[1][0])
        grad_real = self._sum_q1(self.qnn_real_disc.backward([], theta_disc_np)[1][0])
        self.theta_disc.grad.data += torch.tensor(grad_fake-grad_real, dtype=self.theta_disc.dtype)

    def forward_disc(self):
        theta_gen_np,theta_disc_np = self.get_theta_np()
        tmp0 = np.concatenate([theta_disc_np,theta_gen_np])
        prob_fake = self._sum_q1(qiskit.quantum_info.Statevector(self.circ_gen_disc.bind_parameters(tmp0)).probabilities())
        prob_real = self._sum_q1(qiskit.quantum_info.Statevector(self.circ_real_disc.bind_parameters(theta_disc_np)).probabilities())
        return prob_fake,prob_real

    def backward_gen(self):
        theta_gen_np,theta_disc_np = self.get_theta_np()
        grad = -self._sum_q1(self.qnn_gen.backward(theta_disc_np, theta_gen_np)[1][0])
        self.theta_gen.grad.data += torch.tensor(grad, dtype=self.theta_gen.dtype)

    def forward_gen(self):
        theta_gen_np,theta_disc_np = self.get_theta_np()
        tmp0 = np.concatenate([theta_disc_np,theta_gen_np])
        prob_fake = self._sum_q1(qiskit.quantum_info.Statevector(self.circ_gen_disc.bind_parameters(tmp0)).probabilities())
        return prob_fake

    def get_gen_probability(self):
        theta_gen_np = self.theta_gen.detach().numpy()
        ret = qiskit.quantum_info.Statevector(self.circ_gen.bind_parameters(theta_gen_np)).probabilities_dict()
        return ret

circ_real, circ_gen, circ_disc = create_qgan_circuit()

model = QGAN(circ_real, circ_gen, circ_disc)
optimizer_disc = torch.optim.Adam([model.theta_disc], lr=0.02)
optimizer_gen = torch.optim.Adam([model.theta_gen], lr=0.02)
# torch.optim.RMSprop([model.theta_disc], lr=0.0001)

real_prob_dict = qiskit.quantum_info.Statevector(circ_real).probabilities_dict()
print('real:', real_prob_dict)
print('initial generator:', model.get_gen_probability())
# qiskit.visualization.plot_histogram(tmp0, ax=ax0)

num_epoch = 100
num_update_step_disc = 5
history_metric = {'loss_gen':[], 'loss_disc':[], 'kl_div':[]}
best_gen_params = model.theta_gen.detach().numpy().copy()
with tqdm(range(num_epoch)) as pbar:
    for epoch in pbar:
        # Quantum discriminator parameter update
        for ind0 in range(num_update_step_disc):
            optimizer_disc.zero_grad()
            model.backward_disc()
            optimizer_disc.step()
        prob_fake,prob_real = model.forward_disc()
        history_metric['loss_disc'].append(prob_fake - prob_real)

        # Quantum generator parameter update
        optimizer_gen.zero_grad()
        model.backward_gen()
        optimizer_gen.step()
        history_metric['loss_gen'].append(-model.forward_gen())

        # Track KL and save best performing generator weights
        gen_prob_dict = model.get_gen_probability()
        history_metric['kl_div'].append(calculate_KL_divergance(gen_prob_dict, real_prob_dict))
        if history_metric['kl_div'][-1]<=min(history_metric['kl_div']):
            best_gen_params = model.theta_gen.detach().numpy().copy()

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
model.theta_gen.data[:] = torch.tensor(best_gen_params)
print('final generator:', model.get_gen_probability())
# qiskit.visualization.plot_histogram(tmp0, ax=ax0)

