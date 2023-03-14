# jupyter lab required
import numpy as np
import matplotlib.pyplot as plt

import pyzx as zx
import pyzx


circuit = zx.generate.CNOT_HAD_PHASE_circuit(qubits=4,depth=1,clifford=True)
# zx.draw(circuit)
# zx.draw(circuit).savefig('tbd00.png')

circuit.gates

g = circuit.to_graph()
zx.clifford_simp(g)
g.normalize()
zx.draw(g)

c = zx.extract_circuit(g.copy())
zx.draw(c)

c2 = zx.optimize.basic_optimization(c.to_basic_gates())
zx.draw(c2)

zx.compare_tensors(c2, g, preserve_scalar=False)

print(c2.to_qasm())

# c = zx.Circuit.load('circuits/Fast/mod5_4_before')

zx.generate.cliffords(qubits=3, depth=2)
