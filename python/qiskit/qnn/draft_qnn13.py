# https://learn.qiskit.org/course/machine-learning/variational-classification
import functools
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sklearn.preprocessing

import qiskit
import qiskit.algorithms
import qiskit_machine_learning.datasets
import qiskit_machine_learning.algorithms

aer_simulator = qiskit.Aer.get_backend('qasm_simulator')
aer_quantum_instance = qiskit.utils.QuantumInstance(aer_simulator, shots=8192)
aer_sampler = qiskit.opflow.CircuitSampler(aer_quantum_instance)

np_rng = np.random.default_rng()

# qiskit.utils.algorithm_globals.random_seed = 3142
# np.random.seed(qiskit.utils.algorithm_globals.random_seed)

def show_classification_result(xtrain, ytrain, xtest, ytest, prediction):
    assert xtrain.shape[1]==2 and xtest.shape[1]==2
    assert np.all(np.logical_or(ytrain==0, ytrain==1)) and np.all(np.logical_or(ytest==0, ytest==1))

    fig,ax = plt.subplots(figsize=(9, 6))
    for label in [0,1]:
        COLOR = 'C0' if label == 0 else 'C1'
        ind0 = ytrain==label
        ax.scatter(xtrain[ind0,0], xtrain[ind0,1], marker='o', s=100, color=COLOR)
        ind0 = prediction==label
        ax.scatter(xtest[ind0,0], xtest[ind0,1], marker='s', s=100, color=COLOR)
    ind0 = prediction!=ytest
    ax.scatter(xtest[ind0,0], xtest[ind0,1], marker='o', s=500, linewidths=2.5, facecolor='none', edgecolor='C3')

    tmp0 = [
        matplotlib.lines.Line2D([0], [0], marker='o', c='w', mfc='C0', label='label-0', ms=10),
        matplotlib.lines.Line2D([0], [0], marker='o', c='w', mfc='C1', label='label-1', ms=10),
        matplotlib.lines.Line2D([0], [0], marker='s', c='w', mfc='C0', label='predict label-0', ms=10),
        matplotlib.lines.Line2D([0], [0], marker='s', c='w', mfc='C1', label='predict label-1', ms=10),
        matplotlib.lines.Line2D([0], [0], marker='o', c='w', mfc='none', mec='C3', label='wrongly classified', mew=2, ms=15)
    ]
    ax.legend(handles=tmp0, bbox_to_anchor=(1, 1), loc='upper left')
    ax.set_title('Training & Test Data')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.tight_layout()
    return fig, ax

def show_train_loss(loss_list):
    fig,ax = plt.subplots()
    ax.plot(loss_list)
    ax.set_xlabel('step')
    ax.set_ylabel('loss')
    return fig,ax

class OptimizerLog:
    def __init__(self, print_freq=-1):
        self.costs = []
        self.print_freq = print_freq
    def update(self, ind_step, theta, ftheta, _stepsize, _accept):
        if (self.print_freq>0) and (ind_step%self.print_freq==0):
            print(f'[{ind_step}] loss={ftheta}')
        self.costs.append(ftheta)

@functools.lru_cache
def get_parity(num_qubit):
    # Returns 1 if parity of `bitstring` is even, otherwise 0
    tmp0 = [''.join(x) for x in itertools.product(*(['01']*num_qubit))]
    parity_map = {x:(sum(y=='1' for y in x)%2) for x in tmp0}
    parity_key_sorted = sorted(parity_map.keys())
    tmp0 = np.array([parity_map[x] for x in parity_key_sorted])
    parity_index = np.stack([tmp0==0,tmp0==1], axis=0)
    return parity_key_sorted,parity_index

def classification_probability(data, theta):
    circuit_list = []
    tmp0 = {p:theta[i] for i, p in enumerate(VAR_FORM.ordered_parameters)}
    for data_i in data:
        tmp1 = {p:data_i[i] for i,p in enumerate(FEATURE_MAP.ordered_parameters)}
        circuit_list.append(AD_HOC_CIRCUIT.assign_parameters(tmp0 | tmp1))
    parity_key_sorted,parity_index = get_parity(AD_HOC_CIRCUIT.num_qubits)
    results = qiskit.execute(circuit_list, aer_simulator).result()
    tmp0 = [results.get_counts(x) for x in circuit_list]
    tmp1 = np.array([[x.get(y,0) for y in parity_key_sorted] for x in tmp0])
    probability = tmp1[:,parity_index[1]].sum(axis=1)/tmp1.sum(axis=1) #to be label-1
    return probability

hf_binary_cross_entropy = lambda prob, label: -np.log(np.maximum(1e-10, np.where(label==1, prob, 1-prob))).mean()


TRAIN_DATA, TRAIN_LABELS, TEST_DATA, TEST_LABELS = qiskit_machine_learning.datasets.ad_hoc_data(
            training_size=20, test_size=5, n=2, gap=0.3, one_hot=False)


FEATURE_MAP = qiskit.circuit.library.ZZFeatureMap(feature_dimension=2, reps=2)
VAR_FORM = qiskit.circuit.library.TwoLocal(2, ['ry', 'rz'], 'cz', reps=2)
AD_HOC_CIRCUIT = FEATURE_MAP.compose(VAR_FORM)
AD_HOC_CIRCUIT.measure_all()

theta0 = np_rng.uniform(size=VAR_FORM.num_parameters)

log = OptimizerLog(print_freq=15)
optimizer = qiskit.algorithms.optimizers.SPSA(maxiter=100, callback=log.update)
hf0 = lambda theta: hf_binary_cross_entropy(classification_probability(TRAIN_DATA, theta), TRAIN_LABELS)
theta_optim = optimizer.minimize(hf0, theta0) #slow

probability = classification_probability(TEST_DATA, theta_optim.x)
prediction = probability>0.5
test_acc = np.mean(prediction==TEST_LABELS)

fig,ax = show_train_loss(log.costs)
fig,ax = show_classification_result(TRAIN_DATA, TRAIN_LABELS, TEST_DATA, TEST_LABELS, prediction)


encoder = sklearn.preprocessing.OneHotEncoder()
train_labels_oh = encoder.fit_transform(TRAIN_LABELS.reshape(-1, 1)).toarray()
test_labels_oh = encoder.fit_transform(TEST_LABELS.reshape(-1, 1)).toarray()

log = OptimizerLog(print_freq=15)
tmp0 = qiskit.algorithms.optimizers.SPSA(callback=log.update)
# TODO quantum_instance= is depracated
vqc = qiskit_machine_learning.algorithms.classifiers.VQC(feature_map=FEATURE_MAP,
        ansatz=VAR_FORM, loss='cross_entropy', optimizer=tmp0, initial_point=theta0, quantum_instance=aer_quantum_instance)
vqc.fit(TRAIN_DATA, train_labels_oh)

predict_vqc = np.argmax(vqc.predict(TEST_DATA), axis=1)
acc_test = vqc.score(TEST_DATA, test_labels_oh)

fig,ax = show_train_loss(log.costs)
fig,ax = show_classification_result(TRAIN_DATA, TRAIN_LABELS, TEST_DATA, TEST_LABELS, predict_vqc)
