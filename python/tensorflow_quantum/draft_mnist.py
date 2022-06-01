# https://www.tensorflow.org/quantum/tutorials/mnist
import cirq
import sympy
import numpy as np
import collections
import tensorflow as tf
import tensorflow_quantum as tfq

def remove_contradicting(xs, ys):
    # https://arxiv.org/abs/1802.06002 section-3.3
    mapping = collections.defaultdict(set)
    orig_x = dict()
    for x,y in zip(xs,ys):
        tmp0 = tuple(x.flatten())
        orig_x[tmp0] = x
        mapping[tmp0].add(y)
    new_x = []
    new_y = []
    for flatten_x in mapping:
        labels = mapping[flatten_x]
        if len(labels) == 1: # Throw out images that match more than one label.
            new_x.append(orig_x[flatten_x])
            new_y.append(next(iter(labels)))
    num_uniq_3 = sum(1 for value in mapping.values() if len(value) == 1 and True in value)
    num_uniq_6 = sum(1 for value in mapping.values() if len(value) == 1 and False in value)
    num_uniq_both = sum(1 for value in mapping.values() if len(value) == 2)
    print("Number of unique images:", len(mapping.values()))
    print("Number of unique 3s: ", num_uniq_3)
    print("Number of unique 6s: ", num_uniq_6)
    print("Number of unique contradicting labels (both 3 and 6): ", num_uniq_both)
    print("Initial number of images: ", len(xs))
    print("Remaining non-contradicting unique images: ", len(new_x))
    return np.array(new_x), np.array(new_y)


def convert_to_circuit(image):
    """Encode truncated classical image into quantum datapoint."""
    values = image.flatten()
    q0 = cirq.GridQubit.rect(4, 4)
    circ = cirq.Circuit()
    for i, value in enumerate(values):
        if value:
            circ.append(cirq.X(q0[i]))
    return circ

def create_quantum_model():
    """Create a QNN model circuit and readout operation to go along with it."""
    q_data = cirq.GridQubit.rect(4, 4)
    q_readout = cirq.GridQubit(-1, -1)
    circ = cirq.Circuit([cirq.X(q_readout), cirq.H(q_readout)])
    for gate,prefix in [(cirq.XX,'xx1'), (cirq.ZZ,'zz1')]:
        for i, q_i in enumerate(q_data):
            tmp0 = sympy.Symbol(f'{prefix}-{i}')
            circ.append(gate(q_i, q_readout)**tmp0)
    circ.append(cirq.H(q_readout))
    op = cirq.Z(q_readout)
    return circ,op


def preprocess_mnist_dataset(x_train, y_train, x_test, y_test):
    def hf0(data, label, tag_remove_contradict=False, threshold=0.5):
        # filter label=3 or label=6
        ind0 = np.logical_or(label==3, label==6)
        data = data[ind0]/255
        label = label[ind0]==3

        data = tf.image.resize(data[:,:,:,np.newaxis], (4,4)).numpy()
        if tag_remove_contradict: #for trainset only
            data,label = remove_contradicting(data, label)
            # Number of unique images: 10387
            # Number of unique 3s:  4912
            # Number of unique 6s:  5426
            # Number of unique contradicting labels (both 3 and 6):  49
            # Initial number of images:  12049
            # Remaining non-contradicting unique images:  10338
        data = (data>threshold).astype(np.float32)
        data_circ = tfq.convert_to_tensor([convert_to_circuit(x) for x in data])
        label_hinge = (2*label - 1).astype(np.float32)
        return data,data_circ,label,label_hinge
    x_train_proc,x_train_tfcirc,y_train_proc,y_train_hinge = hf0(x_train, y_train, True)
    x_test_proc,x_test_tfcirc,y_test_proc,y_test_hinge = hf0(x_test, y_test, False)
    return x_train_proc,x_train_tfcirc,y_train_proc,y_train_hinge,x_test_proc,x_test_tfcirc,y_test_proc,y_test_hinge

def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true==y_pred, tf.float32)
    ret = tf.reduce_mean(result)
    return ret


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
(x_train_proc,x_train_tfcirc,y_train_proc,y_train_hinge,x_test_proc,x_test_tfcirc,
        y_test_proc,y_test_hinge) = preprocess_mnist_dataset(x_train, y_train, x_test, y_test)

model_circuit, model_readout = create_quantum_model()
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(), dtype=tf.string),
    tfq.layers.PQC(model_circuit, model_readout),
])
model.compile(loss=tf.keras.losses.Hinge(), optimizer=tf.keras.optimizers.Adam(), metrics=[hinge_accuracy])
qnn_history = model.fit(x_train_tfcirc, y_train_hinge, batch_size=32,
        epochs=3, verbose=1, validation_data=(x_test_tfcirc, y_test_hinge)) #10minutes
qnn_results = model.evaluate(x_test_tfcirc, y_test_proc)
qnn_accuracy = qnn_results[1]# 0.9017137289047241
# Epoch 1/3
# 324/324 [==============================] - 192s 591ms/step - loss: 0.7464 - hinge_accuracy: 0.6955 - val_loss: 0.4853 - val_hinge_accuracy: 0.7676
# Epoch 2/3
# 324/324 [==============================] - 194s 598ms/step - loss: 0.4378 - hinge_accuracy: 0.7812 - val_loss: 0.4112 - val_hinge_accuracy: 0.8110
# Epoch 3/3
# 324/324 [==============================] - 195s 601ms/step - loss: 0.3757 - hinge_accuracy: 0.8546 - val_loss: 0.3580 - val_hinge_accuracy: 0.9017

## classical NN
def create_fair_classical_model():
    # A simple model based off LeNet from https://keras.io/examples/mnist_cnn/
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=(4,4,1)))
    model.add(tf.keras.layers.Dense(2, activation='relu'))
    model.add(tf.keras.layers.Dense(1))
    return model
model = create_fair_classical_model()
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
model.fit(x_train_proc, y_train_proc, batch_size=128, epochs=20,
          verbose=1, validation_data=(x_test_proc, y_test_proc))
fair_nn_results = model.evaluate(x_test_proc, y_test_proc)
fair_nn_accuracy = fair_nn_results[1] #0.9161585569381714
