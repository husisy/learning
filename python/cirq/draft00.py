import numpy as np
import sympy

import cirq


def build_circuit(h, jr, jc, x_half_turns, h_half_turns, j_half_turns):
    N0 = len(h)
    rot_x = cirq.XPowGate(exponent=x_half_turns)
    rot_z = cirq.ZPowGate(exponent=h_half_turns)
    rot_11 = cirq.CZPowGate(exponent=j_half_turns)
    all_gate = [rot_x(cirq.GridQubit(x,y)) for x in range(N0) for y in range(N0)]
    all_gate += [rot_z(cirq.GridQubit(x,y)) for x,y in zip(*np.nonzero(h==1))]
    for (i,j),value in np.ndenumerate(jr):
        if value==-1:
            all_gate += [
                cirq.X(cirq.GridQubit(i, j)),
                cirq.X(cirq.GridQubit(i+1, j)),
                rot_11(cirq.GridQubit(i, j), cirq.GridQubit(i+1, j)),
                cirq.X(cirq.GridQubit(i, j)),
                cirq.X(cirq.GridQubit(i+1, j)),
            ]
        else:
            all_gate += [rot_11(cirq.GridQubit(i,j), cirq.GridQubit(i+1,j))]
    for (i,j),value in np.ndenumerate(jc):
        if value==-1:
            all_gate += [
                cirq.X(cirq.GridQubit(i, j)),
                cirq.X(cirq.GridQubit(i, j+1)),
                rot_11(cirq.GridQubit(i, j), cirq.GridQubit(i, j+1)),
                cirq.X(cirq.GridQubit(i, j)),
                cirq.X(cirq.GridQubit(i, j+1)),
            ]
        else:
            all_gate += [rot_11(cirq.GridQubit(i, j), cirq.GridQubit(i, j+1))]
    return all_gate


def obj_func(results, h ,jr, jc):
    def hf_energy(bit_sequence):
        assert isinstance(bit_sequence, np.ndarray) and bit_sequence.dtype==np.uint8
        bit_sequence = bit_sequence.astype(np.int).reshape(h.shape)
        x = 1 - 2*bit_sequence #False->1, True->-1
        ret = np.sum(x*h) + np.sum(jr*x[:-1]*x[1:]) + np.sum(jc*x[:,:-1]*x[:,1:])
        return ret
    tmp0 = results.histogram(key='x', fold_func=hf_energy)
    ret = sum(k*v for k,v in tmp0.items()) / results.repetitions
    return ret


N0 = 3
h = 2*(np.random.rand(N0,N0)>0.5) - 1
jr = 2*(np.random.rand(N0-1,N0)>0.5) - 1
jc = 2*(np.random.rand(N0,N0-1)>0.5) - 1
qubits = [cirq.GridQubit(i, j) for i in range(N0) for j in range(N0)]

simulator = cirq.Simulator()

circuit = cirq.Circuit()
circuit.append(build_circuit(h, jr, jc, 0.1, 0.2, 0.3))
circuit.append(cirq.measure(*qubits, key='x'))
results = simulator.run(circuit, repetitions=10000)
print(results.histogram(key='x')) #collections.Counter
print('Value of the objective function {}'.format(obj_func(results, h ,jr, jc)))

circuit = cirq.Circuit()
alpha = sympy.Symbol('alpha')
beta = sympy.Symbol('beta')
gamma = sympy.Symbol('gamma')
circuit.append(build_circuit(h, jr, jc, alpha, beta, gamma))
circuit.append(cirq.measure(*qubits, key='x'))
resolver = cirq.ParamResolver({'alpha': 0.1, 'beta': 0.3, 'gamma': 0.7})
resolved_circuit = cirq.resolve_parameters(circuit, resolver)

sweep = (cirq.Linspace(key='alpha', start=0.1, stop=0.9, length=5)
         * cirq.Linspace(key='beta', start=0.1, stop=0.9, length=5)
         * cirq.Linspace(key='gamma', start=0.1, stop=0.9, length=5))
results = simulator.run_sweep(circuit, params=sweep, repetitions=100)
[(*list(x.params.param_dict.values()), obj_func(x,h,jr,jc)) for x in results]

# OpenQASM support

# gate
cirq.H, cirq.CNOT, cirq.TOFFOLI, cirq.CCZ
cirq.unitary(cirq.H)
cirq.unitary(cirq.CNOT)
# support gate decomposition

# noise
# cirq.channel()
# TODO density_matrix
# TODO nosie
# TODO measure

# moment
q0 = cirq.GridQubit.rect(1, 3)
z0 = cirq.Moment([cirq.CZ(q0[0], q0[1]), cirq.X(q0[2])])
