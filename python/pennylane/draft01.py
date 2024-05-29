# https://pennylane.ai/qml/demos/tutorial_qubit_rotation.html
# https://pennylane.ai/qml/demos/tutorial_qaoa_intro/
# https://pennylane.ai/qml/demos/tutorial_qaoa_maxcut/
import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
from matplotlib import pyplot as plt
import networkx as nx
from tqdm import tqdm

plt.ion()

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



H = qml.Hamiltonian([1, 1, 0.5], [qml.PauliX(0), qml.PauliZ(1), qml.PauliX(0) @ qml.PauliX(1)])

dev = qml.device('default.qubit', wires=2)

t = 1
n = 2

@qml.qnode(dev)
def circuit():
    qml.ApproxTimeEvolution(H, t, n)
    ret = [qml.expval(qml.PauliZ(i)) for i in range(2)]
    return ret
qml.draw(circuit, expansion_strategy='device')()



def hf0(param):
    qml.RX(param, wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])

@qml.qnode(dev)
def circuit(params, **kwargs):
    qml.layer(hf0, 3, params)
    return [qml.expval(qml.PauliZ(i)) for i in range(2)]

print(qml.draw(circuit)([0.3, 0.4, 0.5]))


edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
graph = nx.Graph(edges)
fig,ax = plt.subplots()
nx.draw(graph, with_labels=True, ax=ax)


cost_h, mixer_h = qml.qaoa.min_vertex_cover(graph, constrained=False)
print("Cost Hamiltonian", cost_h)
print("Mixer Hamiltonian", mixer_h)
cost_h.sparse_matrix().todense() #a diagonal matrix
# https://docs.pennylane.ai/en/stable/code/api/pennylane.qaoa.cost.min_vertex_cover.html

def qaoa_layer(gamma, alpha):
    qml.qaoa.cost_layer(gamma, cost_h)
    qml.qaoa.mixer_layer(alpha, mixer_h)

wires = range(4)
depth = 4 #for a larger depth, more steps are required usually

def circuit(params, **kwargs):
    for w in wires:
        qml.Hadamard(wires=w)
    qml.layer(qaoa_layer, depth, params[0], params[1])

dev = qml.device("default.qubit", wires=wires)

@qml.qnode(dev)
def cost_function(params):
    circuit(params)
    return qml.expval(cost_h)

optimizer = qml.GradientDescentOptimizer()
params = pnp.random.uniform(0, 1, (2, depth), requires_grad=True)
for i in tqdm(range(200)):
    params = optimizer.step(cost_function, params)
print(params)

@qml.qnode(dev)
def probability_circuit(gamma, alpha):
    circuit([gamma, alpha])
    return qml.probs(wires=wires)
probs = probability_circuit(params[0], params[1])
ind0 = (probs>(probs.max().item()*0.99)).nonzero()[0] #(6,10)
print(ind0)

fig,ax = plt.subplots()
ax.bar(range(2**4), probs)



reward_h = qml.qaoa.edge_driver(nx.Graph([(0, 2)]), ['11'])
new_cost_h = cost_h + 2 * reward_h

def qaoa_layer(gamma, alpha):
    qml.qaoa.cost_layer(gamma, new_cost_h)
    qml.qaoa.mixer_layer(alpha, mixer_h)

def circuit(params, **kwargs):
    for w in wires:
        qml.Hadamard(wires=w)
    qml.layer(qaoa_layer, depth, params[0], params[1])

@qml.qnode(dev)
def cost_function(params):
    circuit(params)
    return qml.expval(new_cost_h)
params = pnp.random.uniform(0, 1, (2, depth), requires_grad=True)
for i in tqdm(range(200)):
    params = optimizer.step(cost_function, params)

@qml.qnode(dev)
def probability_circuit(gamma, alpha):
    circuit([gamma, alpha])
    return qml.probs(wires=wires)
probs = probability_circuit(params[0], params[1])
ind0 = (probs>(probs.max().item()*0.99)).nonzero()[0] #(10)
print(ind0)
