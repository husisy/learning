import numpy as np

import cirq


def demo_cirq_measure_bell_state():
    print('# demo_cirq_measure_bell_state')
    num_repeat = 10000
    q0 = cirq.GridQubit.rect(1, 2)
    all_gate = [
        cirq.H(q0[0]),
        cirq.CNOT(q0[0], q0[1]),
        cirq.measure(q0[0], q0[1], key='x'),
    ]
    circuit = cirq.Circuit(all_gate)
    ret0 = cirq.Simulator().run(circuit, repetitions=num_repeat).histogram(key='x') #collection.Counter
    print('|00> counts: {}/{}'.format(ret0[0], num_repeat))
    print('|01> counts: {}/{}'.format(ret0[1], num_repeat))
    print('|10> counts: {}/{}'.format(ret0[2], num_repeat))
    print('|11> counts: {}/{}'.format(ret0[3], num_repeat))


if __name__=='__main__':
    demo_cirq_measure_bell_state()
