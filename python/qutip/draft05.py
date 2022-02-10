'''
Quantum gates and circuits
ref: https://nbviewer.jupyter.org/github/qutip/qutip-notebooks/blob/master/examples/quantum-gates.ipynb#Gates-in-QuTiP-and-their-representation
wiki: https://en.wikipedia.org/wiki/Quantum_logic_gate
'''
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, linewidth=150)
plt.ion()

import qutip
from qutip import rx, ry, rz, cnot, cphase, csign
from qutip import QubitCircuit, gate_sequence_product
from qutip import (berkeley, swap, iswap, swapalpha, sqrtiswap, sqrtswap, sqrtnot, snot, fredkin,
        toffoli, hadamard_transform, phasegate, globalphase)


z0 = QubitCircuit(1, reverse_states=False)
z0.add_gate('RX', targets=[0], arg_value=np.pi/2)
# rx(np.pi/2)


z0 = QubitCircuit(1, reverse_states=False)
z0.add_gate('RY', targets=[0], arg_value=np.pi/2)
# ry(np.pi/2)


z0 = QubitCircuit(1, reverse_states=False)
z0.add_gate('RZ', targets=[0], arg_value=np.pi/2)
# rz(np.pi/2)


z0 = QubitCircuit(2, reverse_states=False)
z0.add_gate('CNOT', controls=[0], targets=[1])
# cnot()


z0 = QubitCircuit(2, reverse_states=False)
z0.add_gate('CSIGN', controls=[0], targets=[1])
# csign()
# cphase(np.pi)


berkeley() #.add_gate('BERKELEY', targets=[0,1])
swap()
iswap()
swapalpha(np.pi/2)
sqrtiswap()
sqrtswap()
sqrtnot()
snot() #hadamard_transform()
phasegate(np.pi/2)
globalphase(np.pi/2)
fredkin()
toffoli()
cnot(N=3)


# a swap gate is equivalent to three CNOT gates
swap()
z0 = QubitCircuit(2)
z0.add_gate('SWAP', targets=[0,1])
gate_sequence_product(z0.propagators())

cnot(control=0, target=1) * cnot(control=1, target=0) * cnot(control=0, target=1)
z1 = QubitCircuit(2)
z1.add_gate('CNOT', targets=1, controls=0)
z1.add_gate('CNOT', targets=0, controls=1)
z1.add_gate('CNOT', targets=1, controls=0)
gate_sequence_product(z1.propagators())

cnot(control=1, target=0) * cnot(control=0, target=1) * cnot(control=1, target=0)
z2 = QubitCircuit(2)
z2.add_gate('CNOT', targets=1, controls=0)
z2.add_gate('CNOT', targets=0, controls=1)
z2.add_gate('CNOT', targets=1, controls=0)
gate_sequence_product(z2.propagators())


## toy exmaple
z0 = QubitCircuit(3)
z0.add_gate('CNOT', targets=1, controls=0)
z0.add_gate('RX', targets=0, arg_value=np.pi/2)
z0.add_gate('RY', targets=1, arg_value=np.pi/2)
z0.add_gate('RZ', targets=2, arg_value=np.pi/2)
z0.add_gate('ISWAP', targets=[1,2])
gate_sequence_product(z0.propagators())

# .resolve_gates() #TODO


'''
Physical implementation of Spin Chain Qubit model
ref: https://nbviewer.jupyter.org/github/qutip/qutip-notebooks/blob/master/examples/spin-chain-model.ipynb
'''
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, linewidth=150)
plt.ion()

import qutip
from qutip.qip.models import circuitprocessor
from qutip.qip.models import spinchain
