import numpy as np
import scipy.linalg
import pyquil

from utils import setup_qvm_quilc_connection
from np_quantum_circuit import np_apply_gate

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
FC, WF_SIM = setup_qvm_quilc_connection()

PauliX = np.array([[0,1],[1,0]])
PauliY = np.array([[0,-1j],[1j,0]])
PauliZ = np.array([[1,0],[0,-1]])

RX_gate = lambda x: np.array([[np.cos(x/2),-1j*np.sin(x/2)], [-1j*np.sin(x/2),np.cos(x/2)]])
RY_gate = lambda x: np.array([[np.cos(x/2), -np.sin(x/2)], [np.sin(x/2),np.cos(x/2)]])
RZ_gate = lambda x: np.array([[np.cos(x/2)-1j*np.sin(x/2),0], [0,np.cos(x/2)+1j*np.sin(x/2)]])


def retrieve_pyquil_state(circuit):
    state_inverse = WF_SIM.wavefunction(pyquil.Program(circuit)).amplitudes
    num_state = state_inverse.size
    assert num_state>1
    num_qubit = round(float(np.log2(num_state)))
    assert num_state==2**num_qubit
    tmp0 = list(range(num_qubit))[::-1]
    ret = state_inverse.reshape([2 for _ in range(num_qubit)]).transpose(*tmp0).reshape(-1)
    return ret


def test_single_gate():
    theta_list = np.random.uniform(0, 2*np.pi, size=(3,))
    ret_ = retrieve_pyquil_state(pyquil.Program(
        pyquil.gates.RX(theta_list[0], 0),
        pyquil.gates.RY(theta_list[1], 0),
        pyquil.gates.RZ(theta_list[2], 0),
    ))
    np_state = np.zeros(2, dtype=np.complex128)
    np_state[0] = 1
    np_state = np_apply_gate(np_state, RX_gate(theta_list[0]), [0])
    np_state = np_apply_gate(np_state, RY_gate(theta_list[1]), [0])
    np_state = np_apply_gate(np_state, RZ_gate(theta_list[2]), [0])
    assert hfe(ret_, np_state) < 1e-7


def test_double_gate(num_qubit=5):
    ind0,ind1 = np.random.permutation(num_qubit)[:2].tolist()
    theta_list = np.random.uniform(0, 2*np.pi, size=(num_qubit,))
    gate_list = [pyquil.gates.RX(y, x) for x,y in enumerate(theta_list)] + [pyquil.gates.CNOT(control=ind0, target=ind1)]
    ret_ = retrieve_pyquil_state(pyquil.Program(gate_list))
    np_state = np.zeros(2**num_qubit, dtype=np.complex128)
    np_state[0] = 1
    for x,y in enumerate(theta_list):
        np_state = np_apply_gate(np_state, RX_gate(y), [x])
    np_state = np_apply_gate(np_state, np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]), [ind0,ind1])
    assert hfe(ret_, np_state) < 1e-7


def pyquil_random_state(num_qubit, num_depth=10, seed=None):
    rand_generator = np.random.RandomState(seed)
    gate_list = []
    for ind0 in range(num_depth):
        tmp0 = rand_generator.uniform(0, 2*np.pi, (num_qubit,3))
        tmp1 = ((pyquil.gates.RX(y0,x),pyquil.gates.RY(y1,x),pyquil.gates.RZ(y2,x)) for x,(y0,y1,y2) in enumerate(tmp0))
        gate_list += [y for x in tmp1 for y in x]
        if ind0%2==0:
            tmp0 = list(range(0,num_qubit-1,2))
        else:
            tmp0 = list(range(num_qubit-(num_qubit//2)*2, num_qubit-1, 2))
        gate_list += [pyquil.gates.CNOT(x,x+1) for x in tmp0]
    circuit = pyquil.Program(gate_list)
    np_state = retrieve_pyquil_state(circuit)
    return circuit, np_state


def np_pyquil_random_state(num_qubit, num_depth=10, seed=None):
    rand_generator = np.random.RandomState(seed)
    CNOT_gate = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
    np_state = np.zeros(2**num_qubit, dtype=np.complex128)
    np_state[0] = 1
    for ind0 in range(num_depth):
        tmp0 = rand_generator.uniform(0, 2*np.pi, (num_qubit,3))
        for x,(y0,y1,y2) in enumerate(tmp0):
            np_state = np_apply_gate(np_state, RX_gate(y0), [x])
            np_state = np_apply_gate(np_state, RY_gate(y1), [x])
            np_state = np_apply_gate(np_state, RZ_gate(y2), [x])
        if ind0%2==0:
            tmp0 = list(range(0,num_qubit-1,2))
        else:
            tmp0 = list(range(num_qubit-(num_qubit//2)*2, num_qubit-1, 2))
        for x in tmp0:
            np_state = np_apply_gate(np_state, CNOT_gate, [x,x+1])
    return np_state


def test_pyquil_random_state(num_qubit=10, num_depth=10):
    seed = np.random.randint(23333)
    _,ret_ = pyquil_random_state(num_qubit, num_depth, seed=seed)
    ret0 = np_pyquil_random_state(num_qubit, num_depth, seed=seed)
    assert hfe(ret_, ret0) < 1e-7


def pyquil_basis_state(ind0, num_qubit=None):
    if num_qubit is None:
        num_qubit = int(np.floor(np.log2(max(1,ind0)))) + 1
    assert ind0>=0 and ind0<2**num_qubit
    tmp0 = bin(ind0)[2:][::-1]
    tmp1 = tmp0 + '0'*(num_qubit-len(tmp0))
    tmp2 = (pyquil.gates.X(x) if y=='1' else pyquil.gates.I(x) for x,y in enumerate(tmp1))
    return pyquil.Program(*tmp2)


def test_pyquil_basis_state(num_qubit=4):
    tmp0 = np.stack([WF_SIM.wavefunction(pyquil_basis_state(x, num_qubit)).amplitudes for x in range(2**num_qubit)])
    assert hfe(np.eye(2**num_qubit), tmp0) < 1e-7


def pyquil_operator_matrix(gate, small_endian=False):
    tmp0 = sorted(gate.get_qubits())
    num_qubit = len(tmp0)
    assert tmp0==list(range(num_qubit))
    ret = []
    for x in range(2**num_qubit):
        circuit = pyquil_basis_state(x, num_qubit)
        circuit += gate
        ret.append(WF_SIM.wavefunction(circuit).amplitudes)
    ret = np.stack(ret, axis=1)
    if small_endian:
        tmp0 = [bin(x)[2:][::-1] for x in range(2**num_qubit)]
        ind0 = [int(x+'0'*(num_qubit-len(x)), base=2) for x in tmp0]
        ret = ret[ind0][:,ind0]
    return ret

def test_pyquil_operator_matrix():
    import pyquil.gates as G
    gate_np_list = [
        (G.I(0), np.array([[1,0],[0,1]])),
        (G.X(0), np.array([[0,1],[1,0]])),
        (G.Y(0), np.array([[0,-1j],[1j,0]])),
        (G.Z(0), np.array([[1,0],[0,-1]])),
        (G.H(0), np.array([[1,1],[1,-1]])/np.sqrt(2)),
        (G.S(0), np.array([[1,0],[0,1j]])),
        (G.T(0), np.array([[1,0],[0,np.exp(1j*np.pi/4)]])),
        (G.CNOT(control=0, target=1), np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])),
    ]

    theta = np.random.randn()
    tmp0 = np.array([[np.cos(theta/2), -1j*np.sin(theta/2)], [-1j*np.sin(theta/2), np.cos(theta/2)]])
    gate_np_list.append((G.RX(theta, qubit=0), tmp0))
    tmp0 = np.array([[np.cos(theta/2), -np.sin(theta/2)], [np.sin(theta/2), np.cos(theta/2)]])
    gate_np_list.append((G.RY(theta, qubit=0), tmp0))
    tmp0 = np.array([[np.exp(-1j*theta/2),0], [0,np.exp(1j*theta/2)]])
    gate_np_list.append((G.RZ(theta, qubit=0), tmp0))
    tmp0 = np.array([[1,0],[0,np.exp(1j*theta)]])
    gate_np_list.append((G.PHASE(theta, qubit=0), tmp0))

    for gate_i,np_i in gate_np_list:
        tmp0 = pyquil_operator_matrix(gate_i, small_endian=True)
        assert hfe(np_i, tmp0)<1e-7


def test_pyquil_measure(num_qubit=3):
    circuit,np0 = pyquil_random_state(num_qubit)
    ro = circuit.declare('ro', memory_type='BIT', memory_size=1)
    circuit += pyquil.gates.MEASURE(0, ro[0])
    np1 = retrieve_pyquil_state(circuit)
    ind0 = np.argmax(np.sum(np.abs(np1.reshape(2,-1))**2, axis=1))
    tmp0 = [np.zeros(2**(num_qubit-1)),np.zeros(2**(num_qubit-1))]
    tmp0[ind0] = np0.reshape(2,-1)[ind0] / np.linalg.norm(np0.reshape(2,-1)[ind0])
    ret_ = np.concatenate(tmp0)
    assert hfe(ret_, np1) < 1e-7


def test_pyquil_paulis(num_qubit=4):
    ind0,ind1 = np.random.permutation(num_qubit)[:2].tolist()
    theta = np.random.rand()
    circuit,np_state = pyquil_random_state(num_qubit)
    pyquil_op = pyquil.paulis.sX(ind0)*pyquil.paulis.sY(ind1)
    circuit += pyquil.paulis.exponential_map(pyquil_op)(theta)
    ret_ = retrieve_pyquil_state(circuit)

    np_op = scipy.linalg.expm(-1j*theta*np.kron(PauliX, PauliY))
    ret0 = np_apply_gate(np_state, np_op, [ind0,ind1])
    assert hfe(ret_, ret0) < 1e-7
