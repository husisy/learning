# https://pennylane.ai/qml/demos/tutorial_qaoa_maxcut/
import  numpy as np
import pennylane as qml
from pennylane import numpy as pnp
import matplotlib.pyplot as plt
plt.ion()

n_wires = 4
graph = [(0, 1), (0, 3), (1, 2), (2, 3)]

dev = qml.device("lightning.qubit", wires=n_wires, shots=1)

@qml.qnode(dev)
def circuit(gammas, betas, ham=None):
    for wire in range(n_wires):
        qml.Hadamard(wires=wire)
    for i in range(len(gammas)):
        for e0,e1 in graph:
            qml.CNOT(wires=[e0, e1])
            qml.RZ(gammas[i], wires=e1)
            qml.CNOT(wires=[e0, e1])
        for wire in range(n_wires):
            qml.RX(2 * betas[i], wires=wire)
    if ham is None:
        ret = qml.sample() #measurement phase
    else:
        ret = qml.expval(ham)
    return ret


def qaoa_maxcut(n_layers=1):
    ham = 0.5*len(graph) - 0.5*sum(qml.PauliZ(e0) @ qml.PauliZ(e1) for e0,e1 in graph)
    # maybe not a good idea to use summation
    def objective(params):
        obj = -circuit(params[0], params[1], ham=ham)
        return obj

    # initialize optimizer: Adagrad works well empirically
    opt = qml.AdagradOptimizer(stepsize=0.5)
    params = 0.01 * pnp.random.rand(2, n_layers, requires_grad=True)
    for i in range(30):
        params = opt.step(objective, params)
        if (i + 1) % 5 == 0:
            tmp0 = np.mean([-objective(params).item() for _ in range(100)])
            print(f"[step={i+1}] obj: {tmp0:.7f}")

    n_samples = 100
    hf0 = lambda x: int(''.join(str(y) for y in x), base=2)
    bit_strings = [hf0(circuit(params[0], params[1], ham=None)) for _ in range(n_samples)]

    counts = pnp.bincount(pnp.array(bit_strings))
    most_freq_bit_string = pnp.argmax(counts)
    print("Optimized (gamma, beta) vectors:\n{}".format(params[:, :n_layers]))
    print("Most frequently sampled bit string is: {:04b}".format(most_freq_bit_string))
    return params, bit_strings


# perform qaoa on our graph with p=1,2 and keep the bitstring sample lists
bitstrings1 = qaoa_maxcut(n_layers=1)[1]
bitstrings2 = qaoa_maxcut(n_layers=2)[1]

xticks = range(0, 16)
xtick_labels = list(map(lambda x: format(x, "04b"), xticks))
bins = pnp.arange(0, 17) - 0.5

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
ax1.set_title("n_layers=1")
ax1.set_xlabel("bitstrings")
ax1.set_ylabel("freq.")
ax1.set_xticks(xticks, xtick_labels, rotation="vertical")
ax1.hist(bitstrings1, bins=bins)
ax2.set_title("n_layers=2")
ax2.set_xlabel("bitstrings")
ax2.set_ylabel("freq.")
ax2.set_xticks(xticks, xtick_labels, rotation="vertical")
ax2.hist(bitstrings2, bins=bins)
fig.tight_layout()
