import time
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))

import qiskit
import qiskit.providers.aer
import qiskit.test.mock
import qiskit.tools.visualization


all_pass = [x for x in dir(qiskit.transpiler.passes) if x[0].isupper()]

qc0 = qiskit.QuantumCircuit(3)
qc0.ccx(0, 1, 2)
pm = qiskit.transpiler.PassManager(qiskit.transpiler.passes.Unroller(['u1', 'u2', 'u3', 'cx']))
qc1 = pm.run(qc0)


# from qiskit.transpiler import CouplingMap, Layout
# from qiskit.transpiler.passes import BasicSwap, LookaheadSwap, StochasticSwap
qc0 = qiskit.QuantumCircuit(7)
qc0.h(3)
qc0.cx(0, 6)
qc0.cx(6, 0)
qc0.cx(0, 1)
qc0.cx(3, 1)
qc0.cx(3, 0)
tmp0 = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
coupling_map = qiskit.transpiler.CouplingMap(couplinglist=tmp0)
transpiler_list = ['BasicSwap', 'LookaheadSwap', 'StochasticSwap']
for transpiler_i in transpiler_list:
    tmp0 = getattr(qiskit.transpiler.passes, transpiler_i)
    pass_manager = qiskit.transpiler.PassManager(tmp0(coupling_map=coupling_map))
    qc1 = pass_manager.run(qc0)
    print(f'{transpiler_i}: depth={qc1.depth()}, gates={qc1.count_ops()}')
# BasicSwap: depth=15, gates=OrderedDict([('swap', 11), ('cx', 5), ('h', 1)])
# LookaheadSwap: depth=9, gates=OrderedDict([('swap', 8), ('cx', 5), ('h', 1)])
# StochasticSwap: depth=10, gates=OrderedDict([('swap', 8), ('cx', 5), ('h', 1)])

backend = qiskit.test.mock.FakeTokyo() # mimics the tokyo device in terms of coupling map and basis gates
init_state = np.zeros(16, dtype=np.complex128)
init_state[0] = 0.5j
init_state[[1,8,15]] = 1/np.sqrt(8)
init_state[9] = 1j/np.sqrt(8)
init_state[14] = 0.5
qc0 = qiskit.QuantumCircuit(10)
qc0.initialize(init_state, range(4))
for level in range(4):
    t0 = time.time()
    qc1 = qiskit.transpile(qc0, backend=backend, seed_transpiler=11, optimization_level=level)
    t1 = time.time()-t0
    print(f'level={level}, time={t1:.3f}, depth={qc1.depth()}, gates={qc1.count_ops()}')
# level=0, time=0.190, depth=73, gates=OrderedDict([('cx', 70), ('u3', 15), ('u1', 15), ('reset', 4)])
# level=1, time=0.122, depth=59, gates=OrderedDict([('cx', 52), ('u3', 15), ('u1', 6)])
# level=2, time=0.104, depth=38, gates=OrderedDict([('cx', 20), ('u3', 15), ('u1', 6)])
# level=3, time=0.155, depth=38, gates=OrderedDict([('cx', 20), ('u3', 15), ('u1', 6)])



# https://qiskit.org/documentation/tutorials/circuits_advanced/04_transpiler_passes_and_passmanager.html
class BasicSwap(qiskit.transpiler.basepasses.TransformationPass):
    def __init__(self, coupling_map, initial_layout=None):
        super().__init__()
        self.coupling_map = coupling_map
        self.initial_layout = initial_layout

    def run(self, dag):
        new_dag = qiskit.dagcircuit.DAGCircuit()
        for qreg in dag.qregs.values():
            new_dag.add_qreg(qreg)
        for creg in dag.cregs.values():
            new_dag.add_creg(creg)

        if self.initial_layout is None:
            if self.property_set["layout"]:
                self.initial_layout = self.property_set["layout"]
            else:
                self.initial_layout = qiskit.transpiler.Layout.generate_trivial_layout(*dag.qregs.values())

        if len(dag.qubits) != len(self.initial_layout):
            raise qiskit.transpiler.TranspilerError('The layout does not match the amount of qubits in the DAG')

        if len(self.coupling_map.physical_qubits) != len(self.initial_layout):
            raise qiskit.transpiler.TranspilerError("Mappers require to have the layout to be the same size as the coupling map")

        canonical_register = dag.qregs['q']
        trivial_layout = qiskit.transpiler.Layout.generate_trivial_layout(canonical_register)
        current_layout = trivial_layout.copy()

        for layer in dag.serial_layers():
            subdag = layer['graph']
            for gate in subdag.two_qubit_ops():
                physical_q0 = current_layout[gate.qargs[0]]
                physical_q1 = current_layout[gate.qargs[1]]
                if self.coupling_map.distance(physical_q0, physical_q1) != 1:
                    # Insert a new layer with the SWAP(s).
                    swap_layer = qiskit.dagcircuit.DAGCircuit()
                    swap_layer.add_qreg(canonical_register)

                    path = self.coupling_map.shortest_undirected_path(physical_q0, physical_q1)
                    for swap in range(len(path) - 2):
                        connected_wire_1 = path[swap]
                        connected_wire_2 = path[swap + 1]

                        qubit_1 = current_layout[connected_wire_1]
                        qubit_2 = current_layout[connected_wire_2]

                        # create the swap operation
                        swap_layer.apply_operation_back(qiskit.circuit.library.SwapGate(), qargs=[qubit_1, qubit_2], cargs=[])

                    # layer insertion
                    order = current_layout.reorder_bits(new_dag.qubits)
                    new_dag.compose(swap_layer, qubits=order)

                    # update current_layout
                    for swap in range(len(path) - 2):
                        current_layout.swap(path[swap], path[swap + 1])
            order = current_layout.reorder_bits(new_dag.qubits)
            new_dag.compose(subdag, qubits=order)
        return new_dag

q = qiskit.QuantumRegister(7, 'q')
qc0 = qiskit.QuantumCircuit(q)
qc0.h(q[0])
qc0.cx(q[0], q[4])
qc0.cx(q[2], q[3])
qc0.cx(q[6], q[1])
qc0.cx(q[5], q[0])
qc0.rz(0.1, q[2])
qc0.cx(q[5], q[0])
tmp0 = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
coupling_map = qiskit.transpiler.CouplingMap(couplinglist=tmp0)
pm = qiskit.transpiler.PassManager(BasicSwap(coupling_map))
qc1 = pm.run(qc0)
