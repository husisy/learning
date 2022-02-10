'''https://quimb.readthedocs.io/en/latest/examples/ex_quantum_circuit.html'''
import numpy as np
import quimb as qu
import matplotlib.pyplot as plt
plt.ion()

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

circ = qu.tensor.Circuit(N=10, tags='PSI0')

for i in range(10):
    circ.apply_gate('H', i, gate_round=0)
for r in range(1, 9):
    for i in range(0, 10, 2):
        circ.apply_gate('CNOT', i, i + 1, gate_round=r)
    for i in range(10):
        circ.apply_gate('RY', 1.234, i, gate_round=r)
    for i in range(1, 9, 2):
        circ.apply_gate('CZ', i, i + 1, gate_round=r)
for i in range(10):
    circ.apply_gate('H', i, gate_round=r+1)

circ.psi.graph(color=['PSI0','H','CNOT','CZ'])
circ.psi.graph(color=[f'I{i}' for i in range(10)])
circ.psi.graph(color=['PSI0'] + [f'ROUND_{i}' for i in range(10)])
