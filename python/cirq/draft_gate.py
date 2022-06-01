import numpy as np
import sympy

import cirq

cirq_sim = cirq.Simulator()


# gate
cirq.H, cirq.CNOT, cirq.TOFFOLI, cirq.CCZ
cirq.unitary(cirq.H)
cirq.unitary(cirq.CNOT)
# support gate decomposition


## custom gate
class MySingleGate00(cirq.Gate):
    # cirq.SingleQubitGate
    def __init__(self):
        super().__init__()
    def _num_qubits_(self):
        return 1
    def _unitary_(self):
        s2 = 1/np.sqrt(2)
        ret = np.array([[s2,s2], [-s2,s2]])
        return ret
    def _circuit_diagram_info_(self, args):
        return "G"

class MyDoubleGate00(cirq.Gate):
    def __init__(self):
        super().__init__()
    def _num_qubits_(self):
        return 2
    def _unitary_(self):
        s2 = 1/np.sqrt(2)
        ret = np.array([[s2,-s2,0,0], [0,0,s2,s2], [s2,s2,0,0], [0,0,s2,-s2]])
        return ret
    def _circuit_diagram_info_(self, args):
        return "Top wire symbol", "Bottom wire symbol"

q0 = cirq.GridQubit(0, 0)
my_gate = MySingleGate00()
circ = cirq.Circuit(my_gate.on(q0))
cirq_sim.simulate(circ).final_state_vector

q0 = cirq.GridQubit(0, 0)
q1 = cirq.GridQubit(0, 1)
my_gate = MyDoubleGate00()
circ = cirq.Circuit(my_gate.on(q0, q1))
cirq_sim.simulate(circ).final_state_vector


class MyParaSingleGate00(cirq.Gate):
    def __init__(self, theta):
        super().__init__()
        self.theta = theta
    def _num_qubits_(self):
        return 1
    def _unitary_(self):
        ct = np.cos(self.theta)
        st = np.sin(self.theta)
        ret = np.array([[ct,st], [st,-ct]])
        return ret
    def _circuit_diagram_info_(self, args):
        return f"R({self.theta})"
q0 = cirq.GridQubit(0, 0)
circ = cirq.Circuit(MyParaSingleGate00(theta=0.1).on(q0))
cirq_sim.simulate(circ).final_state_vector


class MyDoubleGate01(cirq.Gate):
    # SWAP
    def __init__(self):
        super().__init__()
    def _num_qubits_(self):
        return 2
    def _decompose_(self, qubits):
        a, b = qubits
        yield cirq.CNOT(a, b)
        yield cirq.CNOT(b, a)
        yield cirq.CNOT(a, b)
    def _circuit_diagram_info_(self, args):
        return ["CustomSWAP"] * self.num_qubits()
q0 = cirq.GridQubit(0, 0)
q1 = cirq.GridQubit(0, 1)
my_gate = MyDoubleGate01()
circ = cirq.Circuit(cirq.X(q0), my_gate.on(q0,q1))
cirq_sim.simulate(circ).final_state_vector
