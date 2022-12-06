import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets
import sklearn.svm

import qiskit
import qiskit.opflow
import qiskit_machine_learning.datasets
import qiskit_machine_learning.kernels

aer_simulator = qiskit.Aer.get_backend('qasm_simulator')

# qiskit.utils.algorithm_globals.random_seed = 12345

def calculate_kernel(feature_map, x_data, y_data=None):
    if y_data is None:
        y_data = x_data
    x_circuits = qiskit.opflow.CircuitStateFn(feature_map).bind_parameters(
            dict(zip(feature_map.parameters, x_data.T.tolist())))
    y_circuits = qiskit.opflow.CircuitStateFn(feature_map).bind_parameters(
            dict(zip(feature_map.parameters, y_data.T.tolist())))
    x_kernel = x_circuits.to_matrix_op()
    y_kernel = y_circuits.to_matrix_op()
    ret = np.abs(((~y_kernel) @ x_kernel).eval())**2 #(np,float,(Ny,Nx))
    return ret


train_data, train_labels, test_data, test_labels, sample_total = qiskit_machine_learning.datasets.ad_hoc_data(
                training_size=20, test_size=5, n=2, gap=0.3, include_sample_total=True, one_hot=False)

feature_map = qiskit.circuit.library.ZZFeatureMap(feature_dimension=2, reps=2)
train_kernel = calculate_kernel(feature_map, train_data)
test_kernel = calculate_kernel(feature_map, train_data, test_data)

model = sklearn.svm.SVC(kernel='precomputed')
model.fit(train_kernel, train_labels)
test_acc = model.score(test_kernel, test_labels)
print("Number of support vectors for each class:",model.n_support_)
print("Indices of support vectors:", model.support_)

fig, (ax0,ax1) = plt.subplots(1, 2, figsize=(10,5))
ax0.imshow(train_kernel, interpolation='nearest', origin='upper')
ax0.set_title("train kernel")
ax1.imshow(test_kernel, interpolation='nearest', origin='upper', cmap='Blues')
ax1.set_title("test kernel")

fig,ax = plt.subplots()
ax.set_ylim(0, 2 * np.pi)
ax.set_xlim(0, 2 * np.pi)
ind0 = model.support_[:model.n_support_[0]]
ind1 = model.support_[model.n_support_[0]:]
ax.scatter(train_data[ind0,0], train_data[ind0,1], marker='s', label="A support")
ax.scatter(train_data[ind1,0], train_data[ind1, 1], marker='o', c='C3', label="B support")
ax.legend(loc='upper left', frameon=False)



feature_map = qiskit.circuit.library.ZZFeatureMap(feature_dimension=2, reps=2, entanglement='linear')
adhoc_kernel = qiskit_machine_learning.kernels.QuantumKernel(feature_map=feature_map, quantum_instance=aer_simulator)

adhoc_svc = sklearn.svm.SVC(kernel=adhoc_kernel.evaluate)
adhoc_svc.fit(train_data, train_labels)
test_acc = adhoc_svc.score(test_data, test_labels)
