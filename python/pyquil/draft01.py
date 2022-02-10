import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from scipy.optimize import minimize
import itertools
plt.ion()

import qutip
from pyquil import Program, get_qc, list_quantum_computers
from pyquil.gates import CNOT, H, X, Z, I, RY, MEASURE
from pyquil.api import QVMConnection
from pyquil.paulis import PauliSum, PauliTerm, exponential_map, sZ

from utils import setup_qvm_quilc_connection

FC, WF_SIM = setup_qvm_quilc_connection()


def bloch_density_matrix_to_uvw(density_matrix):
    # see wiki: https://en.wikipedia.org/wiki/Bloch_sphere
    assert density_matrix.ndim==2 and density_matrix.shape==(2,2)
    assert np.max(np.abs(density_matrix - np.conjugate(density_matrix.T))) < 1e-7
    u = 2*float(density_matrix[0,1].real)
    v = 2*float(density_matrix[0,1].imag)
    w = float(density_matrix[0,0].real - density_matrix[1,1].real)
    return u, v, w

def bloch_uvw_to_density_matrix(u, v, w):
    ret = 1/2 * np.array([[1+w,u-1j*v], [u+1j*v,1-w]])
    return ret

def plot_quantum_spin(*wave_function_list):
    density_matrix_list = [np.conjugate(x)[:,np.newaxis]*x for x in wave_function_list]
    uvw_list = [np.array(bloch_density_matrix_to_uvw(x)) for x in density_matrix_list]
    bloch_sphere = qutip.Bloch()
    bloch_sphere.add_vectors(uvw_list)
    bloch_sphere.show()
    bloch_sphere.clear()


def basis_counter(measurement:np.ndarray, include_not_exist:bool=False):
    assert measurement.ndim==2
    num_measurement,num_qubit = measurement.shape
    assert num_qubit < 10
    tmp0 = Counter(tuple(int(y) for y in x) for x in measurement.tolist())
    tmp1 = {''.join(str(x) for x in key):value/num_measurement for key,value in tmp0.items()}
    if include_not_exist:
        tmp0 = [''.join(x) for x in itertools.product(*(['01'] * num_qubit))]
        ret = {x:0 for x in tmp0}
        ret.update(tmp1)
    else:
        ret = tmp1
    return ret

circuit = [
    Program(I(0)),
    Program(H(0)), #Program(RY(np.pi/2, 0))
    Program(X(0), RY(np.pi/2,0)),
]
tmp0 = [WF_SIM.wavefunction(x).amplitudes for x in circuit]
plot_quantum_spin(*tmp0)


QC = get_qc('1q-qvm')
circuit = Program()
ro = circuit.declare('ro', memory_type='BIT', memory_size=1)
circuit = circuit + MEASURE(0, ro[0])
circuit.wrap_in_numshots_loop(100)
executable = QC.compile(circuit)
result = QC.run(executable) #(np,int32,(100,1))
print(basis_counter(result))


QC = get_qc('2q-qvm')
circuit = Program(H(0), CNOT(0,1))
ro = circuit.declare('ro', memory_type='BIT', memory_size=2)
circuit += MEASURE(0, ro[0])
circuit += MEASURE(1, ro[1])
circuit.wrap_in_numshots_loop(100)
executable = QC.compile(circuit)
result = QC.run(executable)
print(basis_counter(result))


QC = get_qc('1q-qvm')
circuit = Program(H(0))
ro = circuit.declare('ro', memory_type='BIT', memory_size=2)
circuit = circuit + MEASURE(0, ro[0])
circuit = circuit + MEASURE(0, ro[1])
circuit.wrap_in_numshots_loop(100)
executable = QC.compile(circuit)
result = QC.run(executable)
print(basis_counter(result))


# QAOA ???
n_qubits = 2
Hm = [PauliTerm('X', i, -1.0) for i in range(n_qubits)]

J = np.array([[0,1],[0,0]]) # weight matrix of the Ising model. Only the coefficient (0,1) is non-zero.
Hc = [PauliTerm('Z',i,-J[i,j]) * PauliTerm('Z',j,1) for i in range(n_qubits) for j in range(n_qubits)]
exp_Hm = [exponential_map(x) for x in Hm]
exp_Hc = [exponential_map(x) for x in Hc]
p = 1
β = np.random.uniform(0, np.pi*2, p)
γ = np.random.uniform(0, np.pi*2, p)
initial_state = Program()
for i in range(n_qubits):
    initial_state += H(i)
def create_circuit(β, γ):
    circuit = Program()
    circuit += initial_state
    for i in range(p):
        for term_exp_Hc in exp_Hc:
            circuit += term_exp_Hc(-β[i])
        for term_exp_Hm in exp_Hm:
            circuit += term_exp_Hm(-γ[i])
    return circuit
def evaluate_circuit(beta_gamma):
    β = beta_gamma[:p]
    γ = beta_gamma[p:]
    circuit = create_circuit(β, γ)
    return qvm.pauli_expectation(circuit, sum(Hc))

qvm = QVMConnection(endpoint=FC.sync_endpoint, compiler_endpoint=FC.compiler_endpoint)
result = minimize(evaluate_circuit, np.concatenate([β, γ]), method='L-BFGS-B')
state = WF_SIM.wavefunction(create_circuit(result['x'][:p], result['x'][p:])) #almost |00> + |11>
