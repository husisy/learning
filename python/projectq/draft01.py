import numpy as np

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))
hf_randc = lambda *size: np.random.randn(*size) + 1j*np.random.rand(*size)

from projectq.ops import All, CNOT, H, Measure, Rz, X, Z
from projectq import MainEngine
from projectq.meta import Dagger, Control

# TODO, how to retrieve wavefunction from projectq
# see link: https://en.wikipedia.org/wiki/Bell_state
# see link: https://projectq.readthedocs.io/en/latest/examples.html#quantum-teleportation

def test_np_quantum_teleportation():
    t_cnot = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]]).reshape(2,2,2,2)
    t_hadamard = np.array([[1,1],[1,-1]])/np.sqrt(2)
    t_not = np.array([[0,1],[1,0]])
    t_z = np.array([[1,0],[0,-1]])
    hf_normalize = lambda x: x / np.linalg.norm(x)

    psi = hf_normalize(np.random.randn(2) + np.random.randn(2)*1j)
    q0 = np.kron(psi, np.array([1,0,0,1])/np.sqrt(2)).reshape(2,2,2)
    q0 = np.einsum(q0, [0,1,2], t_cnot, [3,4,0,1], [3,4,2], optimize=True)
    q0 = np.einsum(q0, [0,1,2], t_hadamard, [3,0], [3,1,2], optimize=True)
    measure = [(0,0),(0,1),(1,0),(1,1)][np.random.randint(0, 4)]
    measure_to_ret = {
        (0,0): hf_normalize(q0[0,0]),
        (0,1): hf_normalize(np.dot(t_not, q0[0,1])),
        (1,0): hf_normalize(np.dot(t_z, q0[1,0])),
        (1,1): hf_normalize(np.dot(t_z, np.dot(t_not, q0[1,1]))),
    }
    assert all(hfe(psi,x)<1e-7 for x in measure_to_ret.values())

# TODO
# with Control(eng, b1)