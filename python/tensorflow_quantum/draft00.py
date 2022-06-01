# https://github.com/tensorflow/quantum/blob/master/docs/tutorials/hello_many_worlds.ipynb
import cirq
import sympy
import numpy as np
import tensorflow as tf

import tensorflow_quantum as tfq

cirq_sim = cirq.Simulator()


sym_a,sym_b = sympy.symbols('a b')
q0, q1 = cirq.GridQubit.rect(1, 2)
circ = cirq.Circuit(
    cirq.rx(sym_a).on(q0),
    cirq.ry(sym_b).on(q1),
    cirq.CNOT(control=q0, target=q1),
)
op0 = cirq.Z(q0)
op1 = 0.5*cirq.Z(q0) + cirq.X(q1)
tfq.convert_to_tensor([circ]) #(tf,string,1)
tfq.convert_to_tensor([op0, op1]) #(tf,string,2)

para_list = np.array(np.random.uniform(0, 2 * np.pi, (5, 2)), dtype=np.float32)
z0 = []
for x in para_list:
    tmp0 = cirq_sim.simulate(circ, cirq.ParamResolver({sym_a:x[0], sym_b:x[1]})).final_state_vector
    z0.append(op0.expectation_from_state_vector(tmp0, {q0:0, q1:1}).real)
z1 = tfq.layers.Expectation()(circ, symbol_names=[sym_a,sym_b], symbol_values=para_list, operators=op0)
assert np.abs(np.array(z0)-z1.numpy()[:,0]).max()<1e-5
