import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import qiskit
import qiskit.opflow
import qiskit.providers.aer
import qiskit_machine_learning.kernels
import qiskit_machine_learning.neural_networks
import qiskit_machine_learning.algorithms

np_rng = np.random.default_rng(233)

import sklearn.datasets
import sklearn.preprocessing
import sklearn.model_selection
# from sklearn.datasets import make_blobs
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from qiskit import BasicAer
# from qiskit.circuit.library import ZFeatureMap
# from qiskit.utils import QuantumInstance, algorithm_globals
# from qiskit_machine_learning.kernels import QuantumKernel
# from qiskit_machine_learning.algorithms import PegasosQSVC

aer_qasm_sim = qiskit.providers.aer.QasmSimulator()
aer_state_sim = qiskit.providers.aer.StatevectorSimulator()
qi_sv = qiskit.utils.QuantumInstance(aer_state_sim)
qi_qasm = qiskit.utils.QuantumInstance(aer_qasm_sim, shots=10)

features, labels = sklearn.datasets.make_blobs(n_samples=20, n_features=2, centers=2, random_state=3, shuffle=True)

features = sklearn.preprocessing.MinMaxScaler(feature_range=(0, np.pi)).fit_transform(features)

train_features, test_features, train_labels, test_labels = sklearn.model_selection.train_test_split(
            features, labels, train_size=15, shuffle=False)


# number of qubits is equal to the number of features
num_qubits = 2

# number of steps performed during the training procedure
tau = 100

# regularization parameter
C = 1000


qiskit.utils.algorithm_globals.random_seed = 12345

pegasos_backend = qiskit.utils.QuantumInstance(aer_state_sim,
        seed_simulator=qiskit.utils.algorithm_globals.random_seed,
        seed_transpiler=qiskit.utils.algorithm_globals.random_seed,
)

feature_map = qiskit.circuit.library.ZFeatureMap(feature_dimension=num_qubits, reps=1)
qkernel = qiskit_machine_learning.kernels.QuantumKernel(feature_map=feature_map, quantum_instance=pegasos_backend)



pegasos_qsvc = qiskit_machine_learning.algorithms.PegasosQSVC(quantum_kernel=qkernel, C=C, num_steps=tau)
pegasos_qsvc.fit(train_features, train_labels)

pegasos_score = pegasos_qsvc.score(test_features, test_labels)
print(f"PegasosQSVC classification test score: {pegasos_score}")


grid_step = 0.2
margin = 0.2
grid_x, grid_y = np.meshgrid(
    np.arange(-margin, np.pi + margin, grid_step), np.arange(-margin, np.pi + margin, grid_step)
)

meshgrid_features = np.column_stack((grid_x.ravel(), grid_y.ravel()))
meshgrid_colors = pegasos_qsvc.predict(meshgrid_features)


fig,ax = plt.subplots(figsize=(5, 5))
meshgrid_colors = meshgrid_colors.reshape(grid_x.shape)
ax.pcolormesh(grid_x, grid_y, meshgrid_colors, cmap="RdBu", shading="auto")
ax.scatter(
    train_features[:, 0][train_labels == 0],
    train_features[:, 1][train_labels == 0],
    marker="s",
    facecolors="w",
    edgecolors="r",
    label="A train",
)
ax.scatter(
    train_features[:, 0][train_labels == 1],
    train_features[:, 1][train_labels == 1],
    marker="o",
    facecolors="w",
    edgecolors="b",
    label="B train",
)
ax.scatter(
    test_features[:, 0][test_labels == 0],
    test_features[:, 1][test_labels == 0],
    marker="s",
    facecolors="r",
    edgecolors="r",
    label="A test",
)
plt.scatter(
    test_features[:, 0][test_labels == 1],
    test_features[:, 1][test_labels == 1],
    marker="o",
    facecolors="b",
    edgecolors="b",
    label="B test",
)
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
ax.set_title("Pegasos Classification")
