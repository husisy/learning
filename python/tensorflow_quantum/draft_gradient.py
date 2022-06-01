# https://github.com/tensorflow/quantum/blob/master/docs/tutorials/gradients.ipynb
import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np

cirq_sim = cirq.Simulator()
tfq_expectation = tfq.layers.Expectation(differentiator=tfq.differentiators.ForwardDifference())
tfq_expectation_bp = tfq.layers.Expectation(differentiator=tfq.differentiators.Adjoint())
tfq_expectation_sampled = tfq.layers.SampledExpectation(differentiator=tfq.differentiators.ForwardDifference())
tfq_expectation_sampled_shift = tfq.layers.SampledExpectation(differentiator=tfq.differentiators.ParameterShift())


def my_expectation(op, alpha, circ, q0):
    tmp0 = cirq_sim.simulate(circ, {'alpha': alpha}).final_state_vector
    ret = op.expectation_from_state_vector(tmp0, {q0: 0}).real
    return ret

q0 = cirq.GridQubit(0, 0)
circ = cirq.Circuit(cirq.Y(q0)**sympy.Symbol('alpha'))
pauli_x = cirq.X(q0)

alpha_i = 0.3
alpha_i_tf = tf.convert_to_tensor([[alpha_i]], dtype=tf.float32)
alpha_batch = np.linspace(0, 5, 200)[:,np.newaxis].astype(np.float32)
alpha_batch_tf = tf.convert_to_tensor(alpha_batch, dtype=tf.float32)

ret_ = np.sin(np.pi * alpha_i)
ret_grad_ = np.pi * np.cos(np.pi * alpha_i)
ret_batch_ = np.sin(np.pi * alpha_batch)
ret_grad_batch_ = np.pi * np.cos(np.pi*alpha_batch)

ret0 = my_expectation(pauli_x, alpha_i, circ, q0)
assert abs(ret_-ret0)<1e-6
ret1 = tfq_expectation(circ, operators=pauli_x, symbol_names=['alpha'], symbol_values=[[alpha_i]])
assert abs(ret1.numpy().item()-ret_) < 1e-6

with tf.GradientTape() as tape:
    tape.watch(alpha_i_tf)
    ret1 = tfq_expectation(circ, operators=pauli_x, symbol_names=['alpha'], symbol_values=alpha_i_tf)
ret_grad1 = tape.gradient(ret1, alpha_i_tf)
assert abs(ret1.numpy().item()-ret_) < 1e-6
assert abs(ret_grad1.numpy().item()-ret_grad_) < 1e-2

with tf.GradientTape() as tape:
    tape.watch(alpha_i_tf)
    ret1 = tfq_expectation_bp(circ, operators=pauli_x, symbol_names=['alpha'], symbol_values=alpha_i_tf)
ret_grad1 = tape.gradient(ret1, alpha_i_tf)
assert abs(ret1.numpy().item()-ret_) < 1e-6
assert abs(ret_grad1.numpy().item()-ret_grad_) < 1e-2

# large variance
ret2 = tfq_expectation_sampled(circ, operators=pauli_x, repetitions=500, symbol_names=['alpha'], symbol_values=[[alpha_i]])
abs(ret2.numpy().item() - ret_) #roughly~0.05

with tf.GradientTape() as tape:
    tape.watch(alpha_batch_tf)
    ret_batch0 = tfq_expectation(circ, operators=pauli_x, symbol_names=['alpha'], symbol_values=alpha_batch_tf)
ret_grad_batch0 = tape.gradient(ret_batch0, alpha_batch_tf)
assert np.abs(ret_batch0.numpy() - ret_batch_).max() < 1e-6
assert np.abs(ret_grad_batch0.numpy() - ret_grad_batch_).max() < 1e-2

# unstable
with tf.GradientTape() as tape:
    tape.watch(alpha_batch_tf)
    ret_batch1 = tfq_expectation_sampled(circ, operators=pauli_x, repetitions=500, symbol_names=['alpha'], symbol_values=alpha_batch_tf)
ret_grad_batch1 = tape.gradient(ret_batch1, alpha_batch_tf)
np.abs(ret_grad_batch1.numpy() - ret_grad_batch_).max() #may larger than 100

# much stable
with tf.GradientTape() as tape:
    tape.watch(alpha_batch_tf)
    ret_batch2 = tfq_expectation_sampled_shift(circ, operators=pauli_x, repetitions=500, symbol_names=['alpha'], symbol_values=alpha_batch_tf)
ret_grad_batch2 = tape.gradient(ret_batch2, alpha_batch_tf)
np.abs(ret_grad_batch2.numpy() - ret_grad_batch_).max()




## custom differentiator
class MyDifferentiator(tfq.differentiators.Differentiator):
    """A Toy differentiator for <Y^alpha | X |Y^alpha>."""

    def __init__(self):
        pass

    def get_gradient_circuits(self, programs, symbol_names, symbol_values):
        """Return circuits to compute gradients for given forward pass circuits.

        Every gradient on a quantum computer can be computed via measurements
        of transformed quantum circuits.  Here, you implement a custom gradient
        for a specific circuit.  For a real differentiator, you will need to
        implement this function in a more general way.  See the differentiator
        implementations in the TFQ library for examples.
        """

        # The two terms in the derivative are the same circuit...
        batch_programs = tf.stack([programs, programs], axis=1)

        # ... with shifted parameter values.
        shift = tf.constant(1/2)
        forward = symbol_values + shift
        backward = symbol_values - shift
        batch_symbol_values = tf.stack([forward, backward], axis=1)

        # Weights are the coefficients of the terms in the derivative.
        num_program_copies = tf.shape(batch_programs)[0]
        batch_weights = tf.tile(tf.constant([[[np.pi/2, -np.pi/2]]]), [num_program_copies, 1, 1])

        # The index map simply says which weights go with which circuits.
        batch_mapper = tf.tile(tf.constant([[[0, 1]]]), [num_program_copies, 1, 1])
        return batch_programs, symbol_names, batch_symbol_values, batch_weights, batch_mapper

custom_dif = MyDifferentiator()
custom_grad_expectation = tfq.layers.Expectation(differentiator=custom_dif)

# Now let's get the gradients with finite diff.
values_tensor = tf.convert_to_tensor(alpha_batch)

with tf.GradientTape() as g:
    g.watch(values_tensor)
    exact_outputs = tfq_expectation(circ, operators=[pauli_x], symbol_names=['alpha'], symbol_values=values_tensor)
analytic_finite_diff_gradients = g.gradient(exact_outputs, values_tensor).numpy()

# Now let's get the gradients with custom diff.
with tf.GradientTape() as g:
    g.watch(values_tensor)
    my_outputs = custom_grad_expectation(circ, operators=[pauli_x], symbol_names=['alpha'], symbol_values=values_tensor)
my_gradients = g.gradient(my_outputs, values_tensor).numpy()

# Create a noisy sample based expectation op.
expectation_sampled = tfq.get_sampled_expectation_op(cirq.DensityMatrixSimulator(noise=cirq.depolarize(0.01)))

# Make it differentiable with your differentiator:
# Remember to refresh the differentiator before attaching the new op
custom_dif.refresh()
differentiable_op = custom_dif.generate_differentiable_op(sampled_op=expectation_sampled)

# Prep op inputs.
circuit_tensor = tfq.convert_to_tensor([circ])
op_tensor = tfq.convert_to_tensor([[pauli_x]])
single_value = tf.convert_to_tensor([[alpha_i]])
num_samples_tensor = tf.convert_to_tensor([[5000]])

with tf.GradientTape() as g:
    g.watch(single_value)
    forward_output = differentiable_op(circuit_tensor, ['alpha'], single_value, op_tensor, num_samples_tensor)
my_gradients = g.gradient(forward_output, single_value)
