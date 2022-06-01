import cirq
import random
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

cirq_sim = cirq.Simulator()
# code from github (keyword may be like QNN or something)

def build_circuit(b, theta, kkey):
    num_qubit = b.size
    q0 = cirq.GridQubit.rect(1, num_qubit)
    q_readout = cirq.GridQubit(0, num_qubit)
    all_gate = [cirq.X(x) for x,y in zip(q0,b) if y==1]
    all_gate.append(cirq.rx(-np.pi/2)(q_readout))
    all_gate += [cirq.ControlledGate(cirq.rx(2*x))(y, q_readout) for x,y in zip(theta, q0)]
    all_gate += [cirq.ZPowGate(exponent=1.5)(q_readout), cirq.H(q_readout)] #measure sigma-Y
    all_gate += [cirq.measure(q_readout, key=kkey)]
    ret = cirq.Circuit(all_gate)
    return ret


def hf_float_mod(x, a, b):
    assert a<b
    tmp0 = x-a
    tmp1 = b-a
    ret = tmp0 - np.floor(tmp0/tmp1)*tmp1 + a
    return ret

np_rng = np.random.default_rng()

learning_rate = 0.2
num_qubit = 5
num_repeat = 1000
num_epoch = 10

theta = np_rng.uniform(-np.pi, np.pi, size=num_qubit)

tmp0 = np_rng.choice(np.arange(1,num_qubit,2))
subset = np.sort(np_rng.permutation(num_qubit)[:tmp0])

all_state = np.array([[int(j) for j in np.binary_repr(i,width=num_qubit)] for i in np.arange(2**num_qubit)])
all_label = 2*((all_state[:,subset].sum(axis=1)%2)==0) - 1 #even->1 odd->-1
ind0 = np_rng.permutation(len(all_state))
N0 = len(all_state)//5 + 1
test_state = all_state[ind0[:N0]]
test_label = all_label[ind0[:N0]]
train_state = all_state[ind0[N0:]]
train_label = all_label[ind0[N0:]]

history_loss = []
history_theta = []
kkey = 'main'
tmp0 = [list(zip(train_state, train_label)) for _ in range(num_epoch)]
for x in tmp0:
    random.shuffle(x)
tmp0 = [y for x in tmp0 for y in x]
for data_i,label_i in tmp0:
    main_circuit = build_circuit(data_i, theta, kkey)

    tmp0 = cirq_sim.run(main_circuit, repetitions=num_repeat).histogram(key=kkey)
    loss = 1 - label_i*(tmp0[0] - tmp0[1])/num_repeat
    history_loss.append(loss)

    dloss = np.zeros_like(theta)
    tmp0 = 1 - 4*np.dot(theta, data_i) / np.pi
    dloss[data_i==1] = 2*label_i*np.cos(np.pi/2* tmp0)

    if np.abs(dloss).sum() > 1e-8:
        gradient = loss/(np.linalg.norm(dloss,2)**2)*dloss
        theta = hf_float_mod(theta - learning_rate*gradient, -np.pi, np.pi)
    history_theta.append(theta)
history_theta = np.stack(history_theta)


fig,ax = plt.subplots()
for i in range(num_qubit):
    ax.plot(history_theta[:,i]/np.pi, label='theta_'+str(i))
ax.legend()
ax.set_xlabel('# of training examples seen')
ax.set_title('Evolution of parameters through training (in units of pi)')
