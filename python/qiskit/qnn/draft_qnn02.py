import os
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits
plt.ion()

import qiskit
import qiskit.providers.aer

# https://github.com/QSciTech-QuantumBC-Workshop/Activity-1.3

cp_tableau = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']

aer_qasm_sim = qiskit.providers.aer.QasmSimulator()
aer_state_sim = qiskit.providers.aer.StatevectorSimulator()
qasm_quantum_instance = qiskit.utils.QuantumInstance(aer_qasm_sim, shots=1000)
sv_quantum_instance = qiskit.utils.QuantumInstance(aer_state_sim)

def plt_show_data(data, label):
    indA = label==0
    indB = label==1
    fig,ax = plt.subplots()
    ax.scatter(data[indA,0], data[indA,1], label='A')
    ax.scatter(data[indB,0], data[indB,1], label='B')
    ax.legend()
    ax.set_xlim(data[:,0].min()-0.1, data[:,0].max()+0.1)
    ax.set_ylim(data[:,1].min()-0.1, data[:,1].max()+0.1)
    ax.set_xlabel('x0')
    ax.set_ylabel('x1')
    ax.grid()
    return fig,ax

def plt_bloch_sphere_figure(state_vec, label):
    state_xyz = statevector_to_xyz(state_vec)

    radius = 0.98
    theta = np.linspace(0, np.pi, 100)
    phi = np.linspace(0, 2*np.pi, 100)
    theta,phi = np.meshgrid(theta, phi, indexing='ij')
    sphere_x = radius*np.sin(theta)*np.cos(phi)
    sphere_y = radius*np.sin(theta)*np.sin(phi)
    sphere_z = radius*np.cos(theta)

    fig = plt.figure(figsize=(9,8))
    ax = mpl_toolkits.mplot3d.Axes3D(fig) #auto_add_to_figure=False
    # fig.add_axes(ax)

    ax.plot_surface(sphere_x, sphere_y, sphere_z, color=cp_tableau[2], linewidth=0, antialiased=True, alpha=0.3)
    tmp0,tmp1 = np.meshgrid(np.linspace(-1,1,100), np.linspace(-1,1,100))
    ax.plot_surface(tmp0, tmp1, 0*tmp0, color=cp_tableau[8], linewidth=0, antialiased=True, alpha=0.5)

    tmp0 = np.linspace(0, 2*np.pi, 100)
    tmp1 = [radius*np.cos(tmp0), radius*np.sin(tmp0), 0*tmp0]
    for x,y,z in [(0,1,2), (1,2,0), (2,1,0)]:
        ax.plot(tmp1[x], tmp1[y], tmp1[z], color='k', linewidth=0.8)
    ax.plot(tmp1[0]/np.sqrt(2), tmp1[0]/np.sqrt(2), tmp1[1], color='k', linewidth=0.8)
    ax.plot(tmp1[0]/np.sqrt(2), -tmp1[0]/np.sqrt(2), tmp1[1], color='k', linewidth=0.8)

    indA = label==0
    indB = label==1
    ax.scatter(state_xyz[indA,0], state_xyz[indA,1], state_xyz[indA,2], s=50, color='blue', label='A')
    ax.scatter(state_xyz[indB,0], state_xyz[indB,1], state_xyz[indB,2], s=50, color='red', label='B')
    ax.legend()
    ax.view_init(30, 45)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    return fig,ax

def build_data_embedding_circuit():
    qc_data = qiskit.circuit.ParameterVector('x', 2)
    qc0 = qiskit.QuantumCircuit(1)
    qc0.ry(qc_data[0],0)
    qc0.rz(qc_data[1],0)
    return qc0, qc_data

def build_rotation_model_circuit():
    qc_data = qiskit.circuit.ParameterVector('x', 2)
    qc_para0 = qiskit.circuit.ParameterVector('m', 2)
    qc0 = qiskit.QuantumCircuit(1)
    qc0.ry(qc_data[0],0)
    qc0.rz(qc_data[1]+qc_para0[0],0)
    qc0.ry(qc_para0[1],0)
    return qc0, qc_data, qc_para0

def build_layered_model_circuit(n_layer=1):
    qc_data = qiskit.circuit.ParameterVector('x', 2)
    qc_para_w = qiskit.circuit.ParameterVector('w', 2*n_layer)
    qc_para_b = qiskit.circuit.ParameterVector('m', 2*n_layer)
    qc0 = qiskit.QuantumCircuit(1)
    qc0.ry(qc_para_w[0]*qc_data[0], 0)
    for l in range(n_layer-1):
        qc0.rz(qc_para_w[2*l+1] * qc_data[1] + qc_para_b[2*l+1], 0)
        qc0.ry(qc_para_w[2*l+2] * qc_data[0] + qc_para_b[2*l+2], 0)
    qc0.rz(qc_para_w[2*n_layer-1] * qc_data[1] + qc_para_b[2*n_layer-1],0) #2*n_layer-1
    qc0.ry(qc_para_b[0],0)
    qc_para_all = list(qc_para_b) + list(qc_para_w)
    return qc0, qc_data, qc_para_all

def embed_data(qc0, para_qc0, xdata):
    qc0_list = [qc0.bind_parameters({para_qc0[0]:x[0], para_qc0[1]:x[1]}) for x in xdata]
    return qc0_list

def circuits_to_statevectors(circuits):
    result = qiskit.execute(circuits, aer_state_sim).result()
    ret = np.array([result.get_statevector(x) for x in range(len(circuits))]) #(np,complex128,(N0,2))
    return ret

def statevector_to_xyz(state_vec):
    phi = np.angle(state_vec[:,1]) - np.angle(state_vec[:,0])
    theta = np.arccos(np.abs(state_vec[:,0])) + np.arcsin(np.abs(state_vec[:,1])) #same theta
    ret = np.zeros((state_vec.shape[0],3))
    ret[:,0] = np.sin(theta) * np.cos(phi)
    ret[:,1] = np.sin(theta) * np.sin(phi)
    ret[:,2] = np.cos(theta)
    return ret

def prepare_all_circuits(qc0, qc_data, qc_para, data_xs, model_values, add_measurements=False):
    # model_value_dict = {p:v for (p,v) in zip(qc_para, model_values)}
    classifier_circuit = qc0.bind_parameters(dict(zip(qc_para, model_values)))
    if add_measurements:
        classifier_circuit.measure_all()
    all_circuits = embed_data(classifier_circuit,qc_data,data_xs)
    return all_circuits

def all_results_to_expectation_values(all_results):
    if all_results.backend_name == 'statevector_simulator':
        all_statevectors = np.array([all_results.get_statevector(x) for x in range(len(all_results.results))])
        probablity = (all_statevectors * all_statevectors.conj()).real
        ret = probablity[:,0] - probablity[:,1] #pauli-z expectation
    else:
        all_counts = all_results.get_counts()
        hf0 = lambda x,y: (x-y)/(x+y)
        ret = np.array([hf0(x.get('0',0),x.get('1')) for x in all_counts])
    return ret

def eval_cost_fct_quadratic(expectation_values,label):
    # label (np,float): 0 or 1
    ret = ((1 - expectation_values*(1-2*label))/2)**2
    return ret


def train_classifier(optimizer,eval_cost_fct,quantum_instance,model_circuit,data_params,model_params,data_xs,data_ys,initial_point):
    """
    model_values [list]: Optimal parameter values found by the optimizer
    loss [float]: Final cost value
    nfev [int]: Number of iteration done by the optimizer
    """
    add_measurements = quantum_instance.backend_name != 'statevector_simulator'

    def cost_function(model_values):
        all_circuits = prepare_all_circuits(model_circuit,data_params,model_params,data_xs,model_values,add_measurements)
        all_results = quantum_instance.execute(all_circuits)
        expectation_values = all_results_to_expectation_values(all_results)
        loss = eval_cost_fct(expectation_values,data_ys).mean()
        return loss

    model_values, loss, nfev = optimizer.optimize(len(model_params), cost_function, initial_point=initial_point)
    return model_values, loss, nfev

def build_linear_model_circuit():
    qc_data = qiskit.circuit.ParameterVector('x', 2)
    qc_para_w = qiskit.circuit.ParameterVector('w', 2)
    qc_para_b = qiskit.circuit.ParameterVector('m', 2)
    qc0 = qiskit.QuantumCircuit(1)
    qc0.ry(qc_para_w[0] * qc_data[0], 0)
    qc0.rz(qc_para_w[1] * qc_data[1] + qc_para_b[1], 0)
    qc0.ry(qc_para_b[0], 0)
    qc_para_all = list(qc_para_b) + list(qc_para_w)
    return qc0, qc_data, qc_para_all

def spsa_optimizer_callback(nb_fct_eval, params, fct_value, stepsize, step_accepted, train_history):
    train_history.append((nb_fct_eval,params,fct_value))
    print(f'evaluations : {nb_fct_eval} loss: {fct_value:0.4f}')

def classify(quantum_instance, model_circuit, model_params, model_values, data_params, data_xs):
    add_measurements = quantum_instance.backend_name != 'statevector_simulator'
    all_circuits = prepare_all_circuits(model_circuit,data_params,model_params,data_xs,model_values,add_measurements)
    all_results = quantum_instance.execute(all_circuits)
    expectation_values = all_results_to_expectation_values(all_results)
    prediction = np.where(expectation_values>0, 0, 1)
    return prediction


import sklearn.datasets
n_samples = 100
data_xs, data_ys = sklearn.datasets.make_moons(n_samples = n_samples,noise=0.1, random_state=0)

n_samples = 60

np.random.seed(0)
X = np.random.random((n_samples,2))
y = (X[:,1] > 0.5).astype(np.int64)
linearly_separable = (X, y)

data_xs, data_ys = linearly_separable
# plt_show_data(data_xs, data_ys)


data_embedding_circuit, data_params = build_data_embedding_circuit()
# data_embedding_circuit.draw('mpl', scale=2)

data_circuits = embed_data(data_embedding_circuit, data_params, data_xs)
statevectors = circuits_to_statevectors(data_circuits)
# state_xyz = statevector_to_xyz(statevectors)
# plt_bloch_sphere_figure(statevectors, data_ys)

# classifier_circuit, data_params, model_params = build_rotation_model_circuit()
# # classifier_circuit.draw('mpl', scale=2)

# classifier_circuit, data_params, model_params = build_layered_model_circuit(n_layer=3)
# # classifier_circuit.draw('mpl', scale=2)

# # classifier_circuit, data_params, model_params = build_rotation_model_circuit()
# classifier_circuit, data_params, model_params = build_layered_model_circuit(n_layers=3)
# all_circuits = prepare_all_circuits(classifier_circuit, data_params, model_params ,data_xs, [1]*len(model_params), add_measurements=False)

# all_circuits = prepare_all_circuits(classifier_circuit, data_params, model_params, data_xs, [1]*len(model_params), add_measurements=True)
# all_results = qasm_quantum_instance.execute(all_circuits)
# print(all_results.get_counts())

# all_circuits = prepare_all_circuits(classifier_circuit, data_params, model_params, data_xs, [1]*len(model_params), add_measurements=False)
# all_results = sv_quantum_instance.execute(all_circuits)
# print(all_results.get_statevector(0))


# model = 'rotation'
model = 'linear'
# model = 'layered'
if model == 'rotation':
    classifier_circuit, data_params, model_params = build_rotation_model_circuit()
    initial_point = [0,0]
elif model == 'linear':
    classifier_circuit, data_params, model_params = build_linear_model_circuit()
    initial_point = [0,0,1,1]
elif model == 'layered':
    n_layers = 4
    classifier_circuit, data_params, model_params = build_layered_model_circuit(n_layers)
    initial_point = [0,0] * n_layers + [1,1] * n_layers


train_history = []
hf0 = lambda n, p, v, ss, sa: spsa_optimizer_callback(n, p, v, ss, sa, train_history)
optimizer = qiskit.algorithms.optimizers.SPSA(maxiter=50, callback=hf0)

model_values, loss, nfev = train_classifier(optimizer, eval_cost_fct_quadratic, sv_quantum_instance,
            classifier_circuit, data_params, model_params, data_xs, data_ys, initial_point)

fig,ax = plt.subplots()
tmp0 = np.array([x[0] for x in train_history])
tmp1 = np.array([x[2] for x in train_history])
ax.plot(tmp0, tmp1)
ax.set_xlabel('step')
ax.set_ylabel('train-loss')

all_circuits = prepare_all_circuits(classifier_circuit,data_params,model_params,data_xs,model_values,add_measurements=False)
statevectors = circuits_to_statevectors(all_circuits)
# plt_bloch_sphere_figure(statevectors, data_ys)


predictions_ys = classify(sv_quantum_instance, classifier_circuit, model_params, model_values, data_params, data_xs)
accuracy = (predictions_ys==data_ys).mean()
# plt_show_data(data_xs, predictions_ys==data_ys)

# the parameter trained on sv_quantum_instance can also used on qasm_quantum_instance if no noise
predictions_ys = classify(qasm_quantum_instance, classifier_circuit, model_params, model_values, data_params, data_xs)
accuracy = (predictions_ys==data_ys).mean()


## IBMQ
import qiskit.providers.ibmq

# qiskit.providers.ibmq.load_account()
with open(os.path.expanduser('~/qiskit_token.txt'), 'r') as fid:
    IBMQ_TOKEN = fid.read().strip()
ibmq_provider = qiskit.providers.ibmq.IBMQ.enable_account(IBMQ_TOKEN, group='open', hub='ibm-q', project='main')

ibmq_jakarta = ibmq_provider.get_backend('ibmq_jakarta')
qiskit.visualization.plot_error_map(ibmq_jakarta)

ibmq_quantum_instance = qiskit.QuantumInstance(ibmq_jakarta,shots=8192,initial_layout=[4,])
predictions_ys = classify(ibmq_quantum_instance,classifier_circuit,model_params,model_values,data_params,data_xs)
accuracy = (predictions_ys==data_ys).mean()
