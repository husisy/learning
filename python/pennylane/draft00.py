import pennylane as qml
# qml.numpy
import numpy as np
import matplotlib.pyplot as plt

np_rng = np.random.default_rng()

device = qml.device('default.qubit', wires=2, shots=1000) #shots=None
@qml.qnode(device)
def hf0(x, y):
    qml.RZ(x, wires=0)
    qml.CNOT(wires=[0,1])
    qml.RY(y, wires=1)
    ret = qml.expval(qml.PauliZ(1))
    return ret
x0 = hf0(np.pi/4, 0.3) #(np,float64)
print(qml.draw(hf0)(np.pi/4, 0.3))
# qml.drawer.use_style('black_white')
fig,ax = qml.draw_mpl(hf0)(np.pi/4, 0.3)
# broadcasting
x1 = hf0(*np_rng.uniform(0, 2*np.pi, size=(2,5))) #(np,float64,(5,))
device.capabilities()['supports_broadcasting']


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
