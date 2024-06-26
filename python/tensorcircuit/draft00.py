import numpy as np
import tensorflow as tf

import tensorcircuit as tc

np_rng = np.random.default_rng()
hf_randc = lambda *size: np_rng.normal(size=size) + 1j*np_rng.normal(size=size)

# K = tc.set_backend("tensorflow")
# tc.set_dtype("complex128")
# tc.set_contractor("greedy")

tc.dtypestr #get current dtype, default=complex64
tc.backend #get current backend, default=numpy
with tc.runtime_backend('tensorflow'):
    with tc.runtime_dtype('complex128'):
        pass


tc.Circuit.sgates #gates without parameters
tc.gates.h #tensorcircuit.gates.GateF
x0 = tc.gates.h() #tensorcircuit.gates.Gate
tc.gates.matrix_for_gate(x0) #np.ndarray
x0.tensor #np.ndarray

num_qubit = 3
circ = tc.Circuit(num_qubit)
for i in range(num_qubit):
    circ.H(i)
    # circ.h(i) #same api
circ.cnot(0, 1)
circ.CNOT(1, 0) #same api
circ_ir = circ.to_qir()
circ_ir[0]['gatef']()
circ_ir[0]['gate']
x0 = circ.state() #np.ndarray
# x0 = circ.wavefunction() #same api
x1 = circ.expectation([tc.gates.x(), [1]]) #np.ndarray
# x1 = circ.expectation_ps(x=[1]) #same api
x2 = circ.expectation([tc.gates.z(), [1]], [tc.gates.z(), [2]]) #Z1*Z2
bitstr0,prob0 = circ.perfect_sampling() #np.ndarray arXiv:1201.3974
bitstr1,prob1 = circ.sample()


with tc.runtime_backend('tensorflow'):
    circ = tc.Circuit(num_qubit)
    for i in range(num_qubit):
        circ.H(i)
    circ.CNOT(0, 1)
    x0 = circ.state() #tf.Tensor


tc.Circuit.vgates #gates with parameters
tc.gates.rx(np.pi)
x0 = tc.gates.rx.f(np.pi) #tensorcircuit.gates.Gate


def get_circuit(theta):
    num_qubit = len(theta)
    circ = tc.Circuit(num_qubit)
    for i in range(num_qubit):
        circ.rx(i, theta=theta[i])
    circ.cnot(0, 1)
    return circ

with tc.runtime_backend('tensorflow'):
    K = tc.backend
    num_qubit = 3
    # theta = K.ones(num_qubit)
    theta = K.implicit_randn(num_qubit)

    circ = get_circuit(theta)
    x0 = circ.state() #tf.Tensor

    circ_ir = circ.to_qir()
    x1 = circ_ir[0]['gate'].tensor

    def hf0(theta):
        circ = get_circuit(theta)
        ret = K.real(circ.expectation([tc.gates.z(), [1]]))
        return ret
    fval0,grad0 = K.value_and_grad(hf0)(theta)
    fval1,grad1 = K.jit(K.value_and_grad(hf0))(theta)


# input state
with tc.runtime_backend('tensorflow'):
    K = tc.backend
    num_qubit = 3
    tmp0 = K.convert_to_tensor(hf_randc(2**num_qubit))
    q0 = tmp0 / K.norm(tmp0)
    circ = tc.Circuit(num_qubit, inputs=q0)
    circ.H(0)
    q1 = circ.state()

num_qubit = 3
circ = tc.Circuit(num_qubit)
circ.any(0, 1, unitary=hf_randc(4,4)) #any matrix, not only unitary
circ.exp1(0, 1, theta=0.233, unitary=tc.gates._zz_matrix) #exp(i*theta*Z1*Z2)
z0 = {x:getattr(tc.gates,x) for x in dir(tc.gates) if x.endswith('_matrix')}
