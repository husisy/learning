# https://pennylane.ai/qml/demos/tutorial_qubit_rotation.html
import pennylane as qml
import numpy as np
from pennylane import numpy as pnp


@qml.qnode(qml.device("default.qubit", wires=1))
def hf_circuit(params):
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    return qml.expval(qml.PauliZ(0))
hf_circuit_grad = qml.grad(hf_circuit, argnum=0)

hf_circuit([0.54, 0.12]) #np.cos(0.54)*np.cos(0.12)
hf_circuit_grad([0.54, 0.12]) #[-np.sin(0.54)*np.cos(0.12), -np.cos(0.54)*np.sin(0.12)]

optimizer = qml.GradientDescentOptimizer(stepsize=0.4)
params = np.array([0.011, 0.012])
for ind0 in range(100):
    params = optimizer.step(hf_circuit, params)
    if (ind0 + 1) % 5 == 0:
        print("Cost after step {:5d}: {: .7f}".format(ind0+1, hf_circuit(params)))
print("Optimized rotation angles: {}".format(params))
