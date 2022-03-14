import numpy as np

import qiskit
import qiskit.providers.aer

from numpysim import np_apply_gate

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
np_rng = np.random.default_rng()

aer_state_sim = qiskit.providers.aer.StatevectorSimulator()


def qiskit_get_state(qc0):
    result = aer_state_sim.run(qiskit.transpile(qc0, aer_state_sim)).result()
    state = np.asarray(result.data(0)["statevector"]) #(np,complex128,2**num_qubit)
    return state

def random_state(num_qubit, num_layer=5, seed=None):
    np_rng = np.random.default_rng(seed)
    qc0 = qiskit.QuantumCircuit(num_qubit)
    for _ in range(num_layer):
        for x in range(num_qubit):
            qc0.u(*np_rng.uniform(0, 2*np.pi, 3), x)
        for x in range(0, num_qubit-1, 2):
            qc0.cnot(x, x+1)
        for x in range(1, num_qubit-1, 2):
            qc0.cnot(x, x+1)
    np0 = qiskit_get_state(qc0)
    return qc0, np0

def test_u3_gate():
    num_qubit = 5
    theta = np_rng.uniform(0, 2*np.pi, 3)
    op0 = qiskit.quantum_info.operators.Operator(qiskit.circuit.library.U3Gate(*theta)).data
    ind0 = int(np_rng.integers(0, num_qubit))
    qc0,np0 = random_state(num_qubit)

    ret_ = np_apply_gate(np0, op0, [ind0])

    qc0.u(*theta, num_qubit-1-ind0)
    ret0 = qiskit_get_state(qc0)
    assert hfe(ret_, ret0) < 1e-7
