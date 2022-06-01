import numpy as np
import scipy.linalg

import cirq

from numpysim import np_apply_gate, np_inner_product_psi0_O_psi1

hfe = lambda x,y,eps=1e-3: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

cirq_sim = cirq.Simulator()

def hf_u3(a, theta, phi):
    ca = np.cos(a)
    sa = np.sin(a)
    ct = np.cos(theta)
    st = np.sin(theta)
    cp = np.cos(phi)
    sp = np.sin(phi)
    tmp0 = ca + 1j*sa*ct
    tmp3 = ca - 1j*sa*ct
    tmp1 = sa*st*(sp + 1j*cp)
    tmp2 = sa*st*(-sp + 1j*cp)
    ret = np.stack([tmp0,tmp1,tmp2,tmp3], axis=-1).reshape(*tmp0.shape, 2, 2)
    return ret


def cirq_random_circuit(num_qubit, num_depth=10, seed=None):
    np_rng = np.random.default_rng(seed)
    q0 = cirq.GridQubit.rect(1, num_qubit)
    all_gate = []
    for ind0 in range(num_depth):
        tmp0 = hf_u3(*np_rng.uniform(0, 2*np.pi, (3,num_qubit)))
        all_gate += [cirq.ops.MatrixGate(x)(y) for x,y in zip(tmp0,q0)]
        if ind0%2==0:
            tmp0 = list(range(0,num_qubit-1,2))
        else:
            tmp0 = list(range(num_qubit-(num_qubit//2)*2, num_qubit-1, 2))
        all_gate += [cirq.ops.CNOT(control=q0[x],target=q0[x+1]) for x in tmp0]
    circuit = cirq.Circuit(all_gate)
    final_state = cirq_sim.simulate(circuit).final_state_vector
    return q0, circuit, final_state


def test_single_gate(num_qubit=5):
    q0,circuit,np0 = cirq_random_circuit(num_qubit)
    op0 = hf_u3(*np.random.uniform(0, 2*np.pi, size=(3,)))
    ind0 = int(np.random.randint(num_qubit, size=()))

    ret_ = np_apply_gate(np0, op0, [ind0])
    circuit.append(cirq.ops.MatrixGate(op0)(q0[ind0]))
    ret0 = cirq.Simulator().simulate(circuit).state_vector().copy()
    assert hfe(ret_, ret0) < 1e-5 #default dtype for cirq is float32/complex64


def test_expectation_from_wavefunction():
    num_qubit = 5
    q0,circuit,np0 = cirq_random_circuit(num_qubit)
    np_op = [(0.5, [(np.array([[1,0],[0,-1]]), 3)]), (1, [(np.array([[0,1],[1,0]]), 2)])]
    cirq_op = 0.5 * cirq.Z(q0[3]) + cirq.X(q0[2])

    ret_ = np_inner_product_psi0_O_psi1(np0, np0, np_op)
    ret0 = cirq_op.expectation_from_state_vector(np0, {y:x for x,y in enumerate(q0)})
    ret1 = cirq_sim.simulate_expectation_values(circuit, observables=[cirq_op])[0]
    assert hfe(ret_, ret0) < 1e-5
    assert hfe(ret_, ret1) < 1e-5


def test_operators_power():
    alpha = np.random.default_rng().uniform(0, 2*np.pi)
    tmp0 = [
        np.array([[0,1],[1,0]]),
        np.array([[0,-1j],[1j,0]]),
        np.array([[1,0],[0,-1]]),
    ]
    tmp1 = [cirq.X, cirq.Y, cirq.Z]
    for x,y in zip(tmp0,tmp1):
        tmp2 = scipy.linalg.fractional_matrix_power(x, alpha)
        tmp3 = cirq.unitary(y**alpha)
        assert hfe(tmp2, tmp3) < 1e-7
