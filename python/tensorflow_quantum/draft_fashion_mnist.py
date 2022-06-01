# https://www.tensorflow.org/quantum/tutorials/quantum_data
# https://arxiv.org/abs/2011.01938
import cirq
import sympy
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
from tqdm import tqdm

np.random.seed(1234)

def filter_03(x, y):
    keep = (y == 0) | (y == 3)
    x, y = x[keep], y[keep]
    y = y == 0
    return x,y

def truncate_x(x_train, x_test, n_components=10):
    """Perform PCA on image dataset keeping the top `n_components` components."""
    n_points_train = tf.gather(tf.shape(x_train), 0)
    n_points_test = tf.gather(tf.shape(x_test), 0)

    # Flatten to 1D
    x_train = tf.reshape(x_train, [n_points_train, -1])
    x_test = tf.reshape(x_test, [n_points_test, -1])

    # Normalize.
    feature_mean = tf.reduce_mean(x_train, axis=0)
    x_train_normalized = x_train - feature_mean
    x_test_normalized = x_test - feature_mean

    _, EVC = tf.linalg.eigh(tf.einsum('ji,jk->ik', x_train_normalized, x_train_normalized))
    ret0 = tf.einsum('ij,jk->ik', x_train_normalized, EVC[:,-n_components:])
    ret1 = tf.einsum('ij,jk->ik', x_test_normalized, EVC[:, -n_components:])
    return ret0,ret1


def preprocess_data(x_train, y_train, x_test, y_test, n_components, n_train, n_test):
    def hf0(data, label):
        ind0 = np.logical_or(label==0, label==3)
        data = data[ind0]/255
        label = label[ind0]==3
        return data, label
    x_train, y_train = hf0(x_train, y_train)
    x_test, y_test = hf0(x_test, y_test)
    # PCA
    tmp0 = x_train.reshape(x_train.shape[0], -1)
    feature_mean = tmp0.mean(axis=0)
    tmp1 = tmp0-feature_mean
    EVC = np.linalg.eigh(tmp1.T @ tmp1)[1][:,(-n_components):]
    x_train = tmp1 @ EVC
    x_test = x_test.reshape(x_test.shape[0], -1) @ EVC



(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train/255 #(np,float64,(60000,28,28))
x_test = x_test/255 #(np,float64,(10000,28,28))

x_train, y_train = filter_03(x_train, y_train) #12000
x_test, y_test = filter_03(x_test, y_test) #2000

DATASET_DIM = 10
x_train, x_test = truncate_x(x_train, x_test, n_components=DATASET_DIM) #(tf,float64,(12000,10))


N_TRAIN = 1000
N_TEST = 200
x_train, x_test = x_train[:N_TRAIN], x_test[:N_TEST]
y_train, y_test = y_train[:N_TRAIN], y_test[:N_TEST]


def prepare_pqk_circuits(qubits, classical_source, n_trotter=10):
    """Prepare the pqk feature circuits around a dataset."""
    n_qubits = len(qubits)

    # Prepare random single qubit rotation wall.
    # Prepare a single qubit X,Y,Z rotation wall on qubits
    rotations = np.random.uniform(-2, 2, size=(n_qubits, 3))
    initial_U = cirq.Circuit()
    for i, qubit in enumerate(qubits):
        for j, gate in enumerate([cirq.X, cirq.Y, cirq.Z]):
            initial_U.append(gate(qubit) ** rotations[i,j])

    # Prepare parametrized V: (XX+YY+ZZ)**alpha
    ref_paulis = [cirq.X(q0)*cirq.X(q1) + cirq.Y(q0)*cirq.Y(q1) + cirq.Z(q0)*cirq.Z(q1) for q0, q1 in zip(qubits, qubits[1:])]
    symbols = list(sympy.symbols('ref_0:'+str(len(ref_paulis))))
    tmp0 = tfq.util.exponential(ref_paulis, symbols)
    exp_circuit = cirq.Circuit([tmp0]*n_trotter)

    tmp0 = tfq.convert_to_tensor([initial_U]*len(classical_source))
    full_circuits = tfq.layers.AddCircuit()(tmp0, append=exp_circuit)
    # Replace placeholders in circuits with values from `classical_source`.
    tmp0 = [str(x) for x in symbols]
    tmp1 = tf.convert_to_tensor(classical_source*(n_qubits/(3*n_trotter)))
    ret = tfq.resolve_parameters(full_circuits, tmp0, tmp1)
    return ret

qubits = cirq.GridQubit.rect(1, DATASET_DIM + 1)
q_x_train_circuits = prepare_pqk_circuits(qubits, x_train)
q_x_test_circuits = prepare_pqk_circuits(qubits, x_test)

def get_pqk_features(qubits, data_batch):
    """Get PQK features based on above construction."""
    batch_size = data_batch.shape[0]
    tmp0 = [[cirq.X(q), cirq.Y(q), cirq.Z(q)] for q in qubits]
    ops = tf.tile(tf.reshape(tfq.convert_to_tensor(tmp0), [1, -1]), [batch_size, 1])
    exp_vals = tfq.layers.Expectation()(data_batch, operators=ops)
    rdm = tf.reshape(exp_vals, [batch_size, len(qubits), -1]) # first order reduced density matrix (1-RDM)
    return rdm

x_train_pqk = get_pqk_features(qubits, q_x_train_circuits) #(1000,11,3)
x_test_pqk = get_pqk_features(qubits, q_x_test_circuits) #(200,11,3)

def get_spectrum(vecs, gamma=1.0):
    """Compute the eigenvalues and eigenvectors of the kernel of datapoints."""
    """Computes d[i][j] = e^ -gamma * (vecs[i] - vecs[j]) ** 2 """
    scaled_gamma = gamma / (tf.cast(tf.gather(tf.shape(vecs), 1), tf.float32) * tf.math.reduce_std(vecs))
    KC_qs = scaled_gamma * tf.einsum('ijk->ij',(vecs[:,None,:] - vecs) ** 2)
    S, V = tf.linalg.eigh(KC_qs)
    S = tf.math.abs(S)
    return S, V

S_pqk, V_pqk = get_spectrum(tf.reshape(tf.concat([x_train_pqk, x_test_pqk], 0), [-1, len(qubits) * 3]))
S_original, V_original = get_spectrum(tf.cast(tf.concat([x_train, x_test], 0), tf.float32), gamma=0.005)


def get_stilted_dataset(S, V, S_2, V_2, lambdav=1.1):
    """Prepare new labels that maximize geometric distance between kernels."""
    S_diag = tf.linalg.diag(S ** 0.5)
    S_2_diag = tf.linalg.diag(S_2 / (S_2 + lambdav) ** 2)
    scaling = S_diag @ tf.transpose(V) @ V_2 @ S_2_diag @ tf.transpose(V_2) @ V @ S_diag

    # Generate new lables using the largest eigenvector.
    _, vecs = tf.linalg.eig(scaling)
    new_labels = tf.math.real(tf.einsum('ij,j->i', tf.cast(V @ S_diag, tf.complex64), vecs[-1])).numpy()
    # Create new labels and add some small amount of noise.
    final_y = new_labels > np.median(new_labels)
    noisy_y = (final_y ^ (np.random.uniform(size=final_y.shape) > 0.95))
    return noisy_y

y_relabel = get_stilted_dataset(S_pqk, V_pqk, S_original, V_original) #slow
y_train_new, y_test_new = y_relabel[:N_TRAIN], y_relabel[N_TRAIN:]

def create_pqk_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(32, activation='sigmoid', input_shape=[len(qubits) * 3,]))
    model.add(tf.keras.layers.Dense(16, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(1))
    return model

pqk_model = create_pqk_model()
pqk_model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.003), metrics=['accuracy'])
tmp0 = tf.reshape(x_train_pqk, [N_TRAIN, -1])
pqk_history = {'accuracy':[], 'val_accuracy':[]}
for _ in tqdm(range(100)): #3minutes(100x10 epochs)
    tmp1 = pqk_model.fit(tmp0, y_train_new, batch_size=32, epochs=10,
            verbose=0, validation_data=(tf.reshape(x_test_pqk, [N_TEST, -1]), y_test_new))
    pqk_history['accuracy'] += list(tmp1.history['accuracy'])
    pqk_history['val_accuracy'] += list(tmp1.history['val_accuracy'])
z0 = pqk_model.predict(tf.reshape(x_test_pqk, [N_TEST, -1]))

## classical model
def create_fair_classical_model():
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(32, activation='sigmoid', input_shape=[DATASET_DIM,]))
    model.add(tf.keras.layers.Dense(16, activation='sigmoid'))
    model.add(tf.keras.layers.Dense(1))
    return model

model = create_fair_classical_model()
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.03), metrics=['accuracy'])
classical_history = {'accuracy':[], 'val_accuracy':[]}
for _ in tqdm(range(100)): #1minute(100x10 epochs)
    tmp0 = model.fit(x_train, y_train_new, batch_size=128, epochs=10,
            verbose=0, validation_data=(x_test, y_test_new))
    classical_history['accuracy'] += list(tmp0.history['accuracy'])
    classical_history['val_accuracy'] += list(tmp0.history['val_accuracy'])
z1 = model.predict(tf.reshape(x_test, [N_TEST, -1]))
# meaningless
#   mean(y_train_new)=0.577
#   mean(y_test_new)=0.135

import matplotlib.pyplot as plt
plt.ion()
fig,ax = plt.subplots(figsize=(10,5))
ax.plot(classical_history['accuracy'], label='accuracy_classical')
ax.plot(classical_history['val_accuracy'], label='val_accuracy_classical')
ax.plot(pqk_history['accuracy'], label='accuracy_quantum')
ax.plot(pqk_history['val_accuracy'], label='val_accuracy_quantum')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.legend()
