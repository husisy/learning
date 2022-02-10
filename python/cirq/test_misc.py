import numpy as np
import scipy.linalg

import cirq

from numpysim import np_apply_gate, U3_gate, np_inner_product_psi0_O_psi1

hfe = lambda x,y,eps=1e-3: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


def cirq_random_qubit(num_qubit, num_depth=10, seed=None):
    rand_generator = np.random.RandomState(seed)
    q0 = cirq.GridQubit.rect(1, num_qubit)
    all_gate = []
    for ind0 in range(num_depth):
        tmp0 = U3_gate(*rand_generator.uniform(0, 2*np.pi, (3,num_qubit)))
        all_gate += [cirq.ops.MatrixGate(x)(y) for x,y in zip(tmp0,q0)]
        if ind0%2==0:
            tmp0 = list(range(0,num_qubit-1,2))
        else:
            tmp0 = list(range(num_qubit-(num_qubit//2)*2, num_qubit-1, 2))
        all_gate += [cirq.ops.CNOT(control=q0[x],target=q0[x+1]) for x in tmp0]
    circuit = cirq.Circuit(all_gate)
    final_state = cirq.Simulator().simulate(circuit).state_vector().copy()
    return q0, circuit, final_state


def test_u3_gate(num_qubit=5):
    qubits,circuit,np0 = cirq_random_qubit(num_qubit)
    operator = U3_gate(*np.random.uniform(0, 2*np.pi, size=(3,)))
    ind0 = int(np.random.randint(num_qubit, size=()))

    ret_ = np_apply_gate(np0, operator, [ind0])
    circuit.append(cirq.ops.MatrixGate(operator)(qubits[ind0]))
    ret0 = cirq.Simulator().simulate(circuit).state_vector().copy()
    assert hfe(ret_, ret0) < 1e-5 #singce default dtype for cirq is complex64


def test_expectation_from_wavefunction():
    num_qubit = 5
    qubits,circuit,np0 = cirq_random_qubit(5)
    operator = U3_gate(*np.random.uniform(0, 2*np.pi, size=(3,)))
    ind0 = int(np.random.randint(num_qubit, size=()))

    cirq_op = 0.5 * cirq.Z(qubits[3]) + cirq.X(qubits[2])
    ret_ = cirq_op.expectation_from_state_vector(np0, {y:x for x,y in enumerate(qubits)})

    np_op = [(0.5, [(np.array([[1,0],[0,-1]]), 3)]), (1, [(np.array([[0,1],[1,0]]), 2)])]
    ret0 = np_inner_product_psi0_O_psi1(np0, np0, np_op)
    assert hfe(ret_, ret0) < 1e-5


def test_operators_power():
    tmp0 = np.random.uniform(1, 2)
    ret_ = cirq.unitary(cirq.X**tmp0)
    ret = scipy.linalg.fractional_matrix_power(np.array([[0,1],[1,0]]), tmp0)
    assert hfe(ret_, ret) < 1e-6
