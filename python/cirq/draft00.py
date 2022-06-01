import numpy as np
import sympy

import cirq
# import cirq.contrib.svg

cirq_sim = cirq.Simulator()
cirq_dm_sim = cirq.DensityMatrixSimulator()


def _dummy_circuit00(q0, q1, with_measure):
    tmp0 = cirq.X**0.5
    circ = cirq.Circuit([
        tmp0(q0),
        tmp0(q1),
        cirq.CZ(q0, q1),
        tmp0(q0),
        tmp0(q1),
    ])
    if with_measure:
        circ.append([cirq.measure(q0, key='q0'), cirq.measure(q1, key='q1')])
        # circ1.append([cirq.measure(q0, q1, key='q')])
    return circ

q0 = cirq.GridQubit(0, 0)
q1 = cirq.GridQubit(1, 0)
# q0 = cirq.GridQubit.rect(1, 2)
circ = _dummy_circuit00(q0, q1, False)
# print(circ)
# cirq.contrib.svg.SVGCircuit(circ) #only valid in jupyter environment
state_vector = cirq_sim.simulate(circ, qubit_order=[q0,q1]).final_state_vector #(np,complex64,4)

circ = _dummy_circuit00(q0, q1, True)
count = cirq_sim.run(circ, repetitions=1000)
q0_result = count.data['q0'].values #(np,int64,1000)
q1_result = count.data['q1'].values #(np,int64,1000)
count.histogram(key='q0') #collection.Counter {1:515, 0:485}

circ = _dummy_circuit00(q0, q1, False)
op0 = cirq.X(q0) * cirq.X(q1)
op1 = cirq.Z(q0) * cirq.Z(q1)
expectation = cirq_sim.simulate_expectation_values(circ, observables=[op0, op1])

circ = cirq.Circuit([
    cirq.H(q0),
    cirq.I(q1),
])
cirq_sim.simulate(circ, qubit_order=[q0,q1]).final_state_vector #[1,0,1,0]
cirq_sim.simulate(circ, qubit_order=[q1,q0]).final_state_vector #[1,1,0,0]

op0 = cirq.X**sympy.Symbol('x')
circ = cirq.Circuit([op0(q0), op0(q1)])
cirq_sim.simulate(circ, cirq.ParamResolver({'x': 0.233}))

circ = cirq.Circuit([op0(q0), op0(q1), cirq.measure(q0, key='q0'), cirq.measure(q1, key='q1')])
tmp0 = [cirq.ParamResolver({'x': y / 2.0}) for y in range(3)]
cirq_sim.run_sweep(circ, params=tmp0, repetitions=2)


circ = cirq.Circuit([cirq.H(q0), cirq.amplitude_damp(0.2)(q0), cirq.CNOT(q0,q1)])
# cirq_dm_sim.run(circ) #if measure
result = cirq_dm_sim.simulate(circ).final_density_matrix
