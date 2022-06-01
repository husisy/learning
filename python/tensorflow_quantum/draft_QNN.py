# https://github.com/tensorflow/quantum/blob/master/docs/tutorials/hello_many_worlds.ipynb
import cirq
import sympy
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import tensorflow_quantum as tfq

cirq_sim = cirq.Simulator()

## example00
class DummyQNN00(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.sym_theta = sympy.symbols('theta_1 theta_2 theta_3')
        q0 = cirq.GridQubit(0, 0)
        self.model_circuit = cirq.Circuit(
            cirq.rz(self.sym_theta[0])(q0),
            cirq.ry(self.sym_theta[1])(q0),
            cirq.rx(self.sym_theta[2])(q0),
        )
        self.dense0 = tf.keras.layers.Dense(10, activation='elu')
        self.dense1 = tf.keras.layers.Dense(3)
        self.expectation = tfq.layers.ControlledPQC(self.model_circuit, operators=cirq.Z(q0))

    @tf.function
    def call(self, circuit_in, command_in):
        #circuit_in(tf,string,1)
        #x(tf,float32,(1,1))
        x = self.dense0(command_in)
        x = self.dense1(x)
        ret = self.expectation([circuit_in, x])
        return ret

def make_dummy00_dataset():
    q0 = cirq.GridQubit(0, 0)
    tmp0 = np.random.uniform(0, 2 * np.pi, 3)
    noisy_circ = cirq.Circuit(
        cirq.rx(tmp0[0])(q0),
        cirq.ry(tmp0[1])(q0),
        cirq.rz(tmp0[2])(q0),
    )
    tmp0 = {
        'circuit': tfq.convert_to_tensor([noisy_circ,noisy_circ]), # (tf,string,2)
        'command': tf.convert_to_tensor(np.array([[0], [1]]), dtype=tf.float32),
        'output': tf.convert_to_tensor(np.array([[1], [-1]]), dtype=tf.float32),
    }
    ret = tf.data.Dataset.from_tensor_slices(tmp0).batch(1)
    return ret

model = DummyQNN00()
dataset = make_dummy00_dataset()
hf_loss = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)

history_loss = []
with tqdm(range(30)) as pbar:
    for ind_epoch in pbar:
        loss_sum = 0
        for data_i in dataset:
            with tf.GradientTape() as tape:
                prediction = model(data_i['circuit'], data_i['command'])
                loss_i = hf_loss(data_i['output'], prediction)
            gradient = tape.gradient(loss_i, model.trainable_variables)
            optimizer.apply_gradients(zip(gradient, model.trainable_variables))
            loss_sum += loss_i.numpy().item()
        history_loss.append(loss_sum)
        pbar.set_postfix(loss=f'{history_loss[-1]:.4f}')
prediction = [model(x['circuit'], x['command']).numpy().item() for x in dataset]



## example01
class DummyQNN01(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.sym_theta = sympy.symbols('theta_1 theta_2 theta_3')
        q0 = cirq.GridQubit(0, 0)
        self.model_circuit = cirq.Circuit(
            cirq.rz(self.sym_theta[0])(q0),
            cirq.ry(self.sym_theta[1])(q0),
            cirq.rx(self.sym_theta[2])(q0),
        )
        self.dense0 = tf.keras.layers.Dense(10, activation='elu')
        self.dense1 = tf.keras.layers.Dense(3)
        self.add_circuit = tfq.layers.AddCircuit()
        self.expectation = tfq.layers.Expectation()

    @tf.function
    def call(self, circuit_in, command_in, operator_in):
        #circuit_in(tf,string,1)
        #command_in(tf,float32,(1,1))
        #operator_in(tf,string,(1,1))
        x = self.dense0(command_in)
        x = self.dense1(x)
        full_circuit = self.add_circuit(circuit_in, append=self.model_circuit)
        ret = self.expectation(full_circuit, symbol_names=self.sym_theta, symbol_values=x, operators=operator_in)
        return ret


def make_dummy01_dataset():
    q0 = cirq.GridQubit(0, 0)
    tmp0 = np.random.uniform(0, 2 * np.pi, 3)
    noisy_circ = cirq.Circuit(
        cirq.rx(tmp0[0])(q0),
        cirq.ry(tmp0[1])(q0),
        cirq.rz(tmp0[2])(q0),
    )
    tmp0 = {
        'circuit': tfq.convert_to_tensor([noisy_circ,noisy_circ]), # (tf,string,2)
        'command': tf.convert_to_tensor(np.array([[0], [1]]), dtype=tf.float32),
        'operator': tfq.convert_to_tensor([[cirq.X(q0)], [cirq.Z(q0)]]),
        'output': tf.convert_to_tensor(np.array([[1], [-1]]), dtype=tf.float32),
    }
    ret = tf.data.Dataset.from_tensor_slices(tmp0).batch(1)
    return ret


model = DummyQNN01()
dataset = make_dummy01_dataset()
hf_loss = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)

history_loss = []
with tqdm(range(30)) as pbar:
    for ind_epoch in pbar:
        loss_sum = 0
        for data_i in dataset:
            with tf.GradientTape() as tape:
                prediction = model(data_i['circuit'], data_i['command'], data_i['operator'])
                loss_i = hf_loss(data_i['output'], prediction)
            gradient = tape.gradient(loss_i, model.trainable_variables)
            optimizer.apply_gradients(zip(gradient, model.trainable_variables))
            loss_sum += loss_i.numpy().item()
        history_loss.append(loss_sum)
        pbar.set_postfix(loss=f'{history_loss[-1]:.4f}')
prediction = [model(x['circuit'], x['command'], x['operator']).numpy().item() for x in dataset]
