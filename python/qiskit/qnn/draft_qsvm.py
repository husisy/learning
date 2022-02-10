# https://github.com/QSciTech-QuantumBC-Workshop/Activity-2.3

import os
os.environ['QISKIT_IN_PARALLEL'] = 'TRUE'

# import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches
plt.ion()

import skimage
import sklearn.datasets
import sklearn.model_selection
import sklearn.svm

import qiskit
import qiskit.providers.aer
import qiskit_machine_learning.circuit.library
import qiskit_machine_learning.kernels

aer_qasm_sim = qiskit.providers.aer.QasmSimulator()
aer_state_sim = qiskit.providers.aer.StatevectorSimulator()
qi_sv = qiskit.utils.QuantumInstance(aer_state_sim)
qi_qasm = qiskit.utils.QuantumInstance(aer_qasm_sim, shots=8092)

n_samples = 100

X, Y = sklearn.datasets.make_circles(n_samples=n_samples, noise=0.05, factor=0.4)
A = X[np.where(Y==0)]
B = X[np.where(Y==1)]
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=0)


def angle_embedding(feature_dim):
    x_params = qiskit.circuit.ParameterVector('x', feature_dim)
    qc = qiskit.QuantumCircuit(feature_dim)
    for i in range(feature_dim):
        qc.rx(x_params[i], i)
    return qc

def custom_embedding(num_qubit, num_layer):
    x_param = qiskit.circuit.ParameterVector('x', feature_dim)
    qc = qiskit.QuantumCircuit(num_qubit)
    for _ in range(num_layer):
        for i in range(num_qubit):
            qc.rx(x_param[i], i)
        for i in range(num_qubit - 1):
            qc.cx(i, i+1)
            qc.barrier()
            qc.rz(x_param[i], i+1)
    return x_param, qc



feature_dim = 2
feature_kind = 'ZZ'
if feature_kind=='angle':
    feature_map = angle_embedding(feature_dim)
elif feature_kind=='ZZ':
    feature_map = qiskit.circuit.library.ZZFeatureMap(feature_dimension=feature_dim, reps=1, entanglement= "linear")
elif feature_kind=='Pauli':
    feature_map = qiskit.circuit.library.PauliFeatureMap(feature_dimension=feature_dim, reps=1, entanglement='linear')
elif feature_kind=='raw':
    feature_map = qiskit_machine_learning.circuit.library.RawFeatureVector(feature_dim)
elif feature_kind=='custom':
    _, feature_map = custom_embedding(feature_dim, num_layer=3)
# tmp0 = feature_map.assign_parameters(x_train[0])
# tmp0.decompose().draw('mpl')
feature_kernel = qiskit_machine_learning.kernels.QuantumKernel(feature_map=feature_map, quantum_instance=qi_qasm)


model = sklearn.svm.SVC(kernel=feature_kernel.evaluate)
model.fit(x_train, y_train)
accuracy_train = model.score(x_train, y_train) #should be near 1.0
accuracy_test = model.score(x_test, y_test) #should be near 1.0
model.n_support_ #number of support vector
model.support_ #Indices of support vectors


#Constructing the inner product circuit for given datapoints and feature map
feature_circuit = feature_kernel.construct_circuit(x_train[0], x_train[1])
job = qiskit.execute(feature_circuit, backend=aer_qasm_sim, shots=8092)
counts = job.result().get_counts(feature_circuit)
transition_amplitude = counts['00']/sum(counts.values())


ind0 = np.argsort(y_train)
ind1 = np.argsort(y_test)
train_kernel = feature_kernel.evaluate(x_vec=x_train, y_vec=x_train)[ind0[:,np.newaxis],ind0]
test_kernel = feature_kernel.evaluate(x_vec=x_test, y_vec=x_train)[ind1[:,np.newaxis],ind0]
fig,(ax0,ax1) = plt.subplots(1, 2, figsize=(10,5))
ax0.imshow(train_kernel, cmap="Blues")
ax0.set_title("Train kernel matrix")
ax1.imshow(test_kernel, cmap="Blues")
ax1.set_title("test-train kernel matrix")


fig,ax = plt.subplots()
tmp0 = x_train[model.support_[:model.n_support_[0]]]
ind0 = np.zeros(y_train.shape, dtype=np.bool_)
ind0[model.support_] = 1
tmp0 = x_train[np.logical_and(y_train==0, np.logical_not(ind0))]
ax.scatter(tmp0[:,0], tmp0[:,1], color='g', marker='s', label="A")
tmp0 = x_train[np.logical_and(y_train==0, ind0)]
ax.scatter(tmp0[:,0], tmp0[:,1], color='r', marker='s', label="A support")
tmp0 = x_train[np.logical_and(y_train==1, np.logical_not(ind0))]
ax.scatter(tmp0[:,0], tmp0[:,1], color='b', label="B")
tmp0 = x_train[np.logical_and(y_train==1, ind0)]
ax.scatter(tmp0[:,0], tmp0[:,1], color='r', label="B support")
ax.legend()



## mnist
# dataset is made up of 1797 8x8 images
mnist = sklearn.datasets.load_digits()
# mnist.images (np,float64,(1797,8,8))
# mnist.target (np,int32,1797)

# Filter digits and targets
labels = [0, 3]
ind0 = np.logical_or(mnist.target==labels[0], mnist.target==labels[1])
x = mnist.images[ind0]
y = mnist.target[ind0]
# We keep only 24 images for each label
class_size = 24
x0 = x[y == labels[0]][:class_size]
x1 = x[y == labels[1]][:class_size]


labels = np.array([0]*class_size + [1]*class_size)
x_train_full_scale, x_test_full_scale, y_train, y_test = sklearn.model_selection.train_test_split(
            np.concatenate((x0, x1), axis=0), labels, test_size=0.2, stratify=labels)

x_train = np.vstack([[skimage.transform.resize(x_i, (4,1), anti_aliasing=False) for x_i in x_train_full_scale]])
x_test = np.vstack([[skimage.transform.resize(x_i, (4,1), anti_aliasing=False) for x_i in x_test_full_scale]])

num_samples = 8
sources = [x_train_full_scale, x_train]
fig, axes = plt.subplots(nrows=2, ncols=num_samples, figsize=(12, 4))
for i in range(num_samples):
    for s in range(len(sources)):
        axes[s, i].imshow(sources[s][i,:,:], cmap=plt.cm.gray_r)
        axes[s, i ].set_xticks([])
        axes[s, i].set_yticks([])
        axes[s, i].set_title(f"Label: {y_train[i]}")
