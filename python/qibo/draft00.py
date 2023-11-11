import numpy as np
import qibo

circ = qibo.models.QFT(nqubits=5)
q0 = circ() #qibo.states.CircuitResult
np0 = np.asarray(q0)

circ = qibo.Circuit(2)
circ.add(qibo.gates.X(0))
circ.add(qibo.gates.M(0, 1)) #measure both qubits
