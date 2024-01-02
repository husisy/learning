import time
import numpy as np
import tensorflow as tf

import qibo

qibo.set_backend('tensorflow')

def get_dummy_callback():
    _state = [0, time.time()]
    def hf0(*args):
        t0 = time.time()
        print(f'[step={_state[0]}] time={t0-_state[1]:.2f}')
        _state[0] = _state[0] + 1
        _state[1] = t0
    return hf0

nqubits = 6
nlayers  = 4
circ = qibo.models.Circuit(nqubits)
for l in range(nlayers):
    circ.add((qibo.gates.RY(q, theta=0) for q in range(nqubits)))
    circ.add((qibo.gates.CZ(q, q+1) for q in range(0, nqubits-1, 2)))
    circ.add((qibo.gates.RY(q, theta=0) for q in range(nqubits)))
    circ.add((qibo.gates.CZ(q, q+1) for q in range(1, nqubits-2, 2)))
    circ.add(qibo.gates.CZ(0, nqubits-1))
circ.add((qibo.gates.RY(q, theta=0) for q in range(nqubits)))
circ = circ.fuse() #usefull when #qubits>12
hamiltonian = qibo.hamiltonians.XXZ(nqubits=nqubits)
vqe = qibo.models.VQE(circ, hamiltonian)
theta0 = np.random.uniform(0, 2*np.pi, 2*nqubits*nlayers + nqubits)
best, params, extra = vqe.minimize(theta0, method='BFGS', compile=False, callback=get_dummy_callback())
# 1.6seconds per iteration, more than 270 iterations
# best: -9.47213494573908


def myloss(param, circ, target):
    circ.set_parameters(param)
    final_state = circ().state()
    ret = 1 - np.abs(np.conj(target).dot(final_state))
    return ret

nqubits = 6
nlayers  = 2

circ = qibo.models.Circuit(nqubits)
for l in range(nlayers):
    circ.add((qibo.gates.RY(q, theta=0) for q in range(nqubits)))
    circ.add((qibo.gates.CZ(q, q+1) for q in range(0, nqubits-1, 2)))
    circ.add((qibo.gates.RY(q, theta=0) for q in range(nqubits)))
    circ.add((qibo.gates.CZ(q, q+1) for q in range(1, nqubits-2, 2)))
    circ.add(qibo.gates.CZ(0, nqubits-1))
circ.add((qibo.gates.RY(q, theta=0) for q in range(nqubits)))

theta0 = np.random.uniform(0, 2*np.pi, 2*nqubits*nlayers + nqubits)
data = np.random.normal(0, 1, size=2**nqubits)
best, params, extra = qibo.optimizers.optimize(myloss, theta0, args=(circ, data), method='BFGS', callback=get_dummy_callback())
circ.set_parameters(params)
