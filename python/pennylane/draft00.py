import pennylane as qml
from pennylane import numpy as np


## Gaussian transformation
# https://pennylane.readthedocs.io/en/latest/tutorials/pennylane_run_gaussian_transformation.html
dev_gaussian = qml.device('default.gaussian', wires=1)
@qml.qnode(dev_gaussian)
def mean_photon_gaussian(magnitude_alpha, phase_alpha, phi):
    qml.Displacement(magnitude_alpha, phase_alpha, wires=0)
    qml.Rotation(phi, wires=0)
    return qml.expval(qml.NumberOperator(0))


def cost(params):
    return (mean_photon_gaussian(params[0], params[1], params[2]) - 1) ** 2

opt = qml.GradientDescentOptimizer(stepsize=0.1)
params = [0.015, 0.02, 0.005]
print('step=0, cost={}'.format(cost(params)))
for ind0 in range(20):
    params = opt.step(cost, params)
    print('step={}, cost={}'.format(ind0+1, cost(params)))


## Strawberry Fields plugin, a non-Gaussian circuit
# https://pennylane.readthedocs.io/en/latest/tutorials/pennylane_run_plugins_hybrid.html
dev_fock = qml.device('strawberryfields.fock', wires=2, cutoff_dim=2)

@qml.qnode(dev_fock)
def photon_redirection(params):
    qml.FockState(1, wires=0)
    qml.Beamsplitter(params[0], params[1], wires=[0,1])
    return qml.expval(qml.NumberOperator(1))

def cost(params):
    return -photon_redirection(params)

opt = qml.GradientDescentOptimizer(stepsize=0.4)
params = [0.01, 0.01]
print('step=0, cost={}'.format(cost(params)))
for ind0 in range(20):
    params = opt.step(cost, params)
    print('step={}, cost={}'.format(ind0+1, cost(params)))


## hybrid computation
dev_qubit = qml.device('default.qubit')
