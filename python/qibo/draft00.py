import numpy as np

import qibo

# qibo.set_backend("qibojit")
qibo.get_metropolis_threshold() #100000
qibo.get_batch_size() #2**18=262144
# qibo.set_precision("double") #single double(default)
# qibo.get_threads()

print(qibo.models.QFT(nqubits=5).draw())
# q0: ─H─U1─U1─U1─U1───────────────────────────x───
# q1: ───o──|──|──|──H─U1─U1─U1────────────────|─x─
# q2: ──────o──|──|────o──|──|──H─U1─U1────────|─|─
# q3: ─────────o──|───────o──|────o──|──H─U1───|─x─
# q4: ────────────o──────────o───────o────o──H─x───

circ = qibo.models.QFT(nqubits=5)
q0 = circ() #qibo.states.CircuitResult
np0 = np.asarray(q0)

circ = qibo.Circuit(2)
circ.add(qibo.gates.X(0))
circ.add(qibo.gates.M(0, 1)) #measure both qubits
# circ.compile() #only needed for backend="tensorflow"
result = circ(nshots=100)
x0 = result.samples(binary=True) #(np,int32,(100,2))
x1 = result.samples(binary=False) #(np,int32,(100,))
x2 = result.frequencies(binary=True) #collections.Counter
x3 = result.frequencies(binary=False) #collections.Counter
q0 = result.state() #(np,complex128,(4,)) before measurement

circ = qibo.Circuit(5)
circ.add(qibo.gates.X(0))
circ.add(qibo.gates.X(4))
# circ.add(qibo.gates.CU1(0, 1, 0.1234))
circ.add(qibo.gates.M(0, 1, register_name="A"))
circ.add(qibo.gates.M(3, 4, register_name="B"))
q0 = np.ones(4) * 0.5 #default |00>
result = circ(q0, nshots=100)
x0 = result.samples(binary=False, registers=True)

circ = qibo.Circuit(3)
circ.add(qibo.gates.H(0))
circ.add(qibo.gates.H(1))
circ.add(qibo.gates.CNOT(0, 2))
circ.add(qibo.gates.CNOT(1, 2))
circ.add(qibo.gates.H(2))
circ.add(qibo.gates.TOFFOLI(0, 1, 2))
print(circ.summary())
'''
Circuit depth = 5
Total number of gates = 6
Number of qubits = 3
Most common gates:
h: 3
cx: 2
ccx: 1
'''
circ.gate_names.most_common()
# [('h', 3), ('cx', 2), ('ccx', 1)]
circ.gates_of_type(qibo.gates.H)


entropy = qibo.callbacks.EntanglementEntropy([0]) #von Neumann entropy for RDM
circ = qibo.models.Circuit(2) # state is |00> (entropy = 0)
circ.add(qibo.gates.CallbackGate(entropy)) # performs entropy calculation in the initial state
circ.add(qibo.gates.H(0)) # state is |+0> (entropy = 0)
circ.add(qibo.gates.CallbackGate(entropy)) # performs entropy calculation after H
circ.add(qibo.gates.CNOT(0, 1)) # state is |00> + |11> (entropy = 1))
circ.add(qibo.gates.CallbackGate(entropy)) # performs entropy calculation after CNOT
q0 = circ()
entropy[0]
entropy[1]
entropy[2]
entropy[:] #0, 0, 1
entropy.nqubits #2


circ = qibo.Circuit(3)
circ.add(qibo.gates.RX(0, theta=0))
circ.add(qibo.gates.RY(1, theta=0))
circ.add(qibo.gates.CZ(1, 2))
circ.add(qibo.gates.fSim(0, 2, theta=0, phi=0))
circ.add(qibo.gates.H(2))
circ.set_parameters([0.123, 0.456, (0.789, 0.321)])

circ = qibo.Circuit(3)
g0 = qibo.gates.RX(0, theta=0)
g1 = qibo.gates.RY(1, theta=0)
g2 = qibo.gates.fSim(0, 2, theta=0, phi=0)
circ.add([g0, g1, qibo.gates.CZ(1, 2), g2, qibo.gates.H(2)])
circ.set_parameters({g0: 0.123, g1: 0.456, g2: (0.789, 0.321)})
# circ.set_parameters([0.123, 0.456, (0.789, 0.321)])
# circ.set_parameters([0.123, 0.456, 0.789, 0.321])

circ = qibo.Circuit(3)
circ.add(qibo.gates.RX(0, theta=0.123))
circ.add(qibo.gates.RY(1, theta=0.456, trainable=False))
circ.add(qibo.gates.fSim(0, 2, theta=0.789, phi=0.567))
x0 = circ.get_parameters() #[(0.123,), (0.789, 0.567)]
# circ.get_parameters(format='list') #list(default) dict flatlist


from qibo import Circuit, gates

circ = qibo.Circuit(1, density_matrix=True)
circ.add(qibo.gates.H(0))
output = circ.add(qibo.gates.M(0, collapse=True))
circ.add(qibo.gates.H(0))
result = circ(nshots=1)
rho0 = result.state() # |+><+| if output=0, |-><-| if output=1


## invert circuit
circ0 = qibo.Circuit(6)
circ0.add([qibo.gates.RX(i, theta=0.1) for i in range(5)])
circ0.add([qibo.gates.CZ(i, i + 1) for i in range(0, 5, 2)])
circ1 = qibo.Circuit(6)
circ1.add([qibo.gates.CU2(i, i + 1, phi=0.1, lam=0.2) for i in range(0, 5, 2)])
circ = circ0 + circ1 + circ0.invert()

## concatenate circuits on part of qubits
circ0 = qibo.Circuit(4)
circ0.add((qibo.gates.RX(i, theta=0.1) for i in range(4)))
circ0.add((qibo.gates.CNOT(0, 1), qibo.gates.CNOT(2, 3)))
circ1 = qibo.Circuit(8)
circ1.add(circ0.on_qubits(*range(0, 8, 2)))
circ1.add(qibo.models.QFT(4).on_qubits(*range(1, 8, 2)))
circ1.add(qibo.models.QFT(6).invert().on_qubits(*range(6)))


## QNN
