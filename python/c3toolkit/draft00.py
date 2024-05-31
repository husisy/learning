import os
import copy
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import qiskit
import qiskit.tools.visualization

import c3
import c3.model
import c3.c3objs
import c3.parametermap
import c3.libraries.chip
import c3.libraries.tasks
import c3.libraries.hamiltonians
import c3.libraries.envelopes
import c3.generator
import c3.generator.devices
import c3.generator.generator
import c3.signal.pulse
import c3.signal.gates
import c3.experiment
import c3.qiskit.c3_gates
import c3.optimizers.optimalcontrol
import c3.libraries.fidelities
import c3.libraries.algorithms

hf_logdir = lambda *x: os.path.join('tbd00', *x)
if not os.path.exists(hf_logdir()):
    os.makedirs(hf_logdir())

qubit_lvls = 3
freq_q1 = 5e9
anhar_q1 = -210e6
t1_q1 = 27e-6
t2star_q1 = 39e-6
qubit_temp = 50e-3

tmp0 = c3.c3objs.Quantity(value=freq_q1, min_val=4.995e9, max_val=5.005e9, unit='Hz 2pi')
tmp1 = c3.c3objs.Quantity(value=anhar_q1, min_val=-380e6, max_val=-120e6, unit='Hz 2pi')
tmp2 = c3.c3objs.Quantity(value=t1_q1, min_val=1e-6, max_val=90e-6, unit='s')
tmp3 = c3.c3objs.Quantity(value=t2star_q1, min_val=10e-6, max_val=90e-3, unit='s')
tmp4 = c3.c3objs.Quantity(value=qubit_temp, min_val=0.0, max_val=0.12, unit='K')
q1 = c3.libraries.chip.Qubit(name="Q1", desc="Qubit 1", freq=tmp0, anhar=tmp1, hilbert_dim=qubit_lvls, t1=tmp2, t2star=tmp3, temp=tmp4)

# the second qubit
freq_q2 = 5.6e9
anhar_q2 = -240e6
t1_q2 = 23e-6
t2star_q2 = 31e-6
tmp0 = c3.c3objs.Quantity(value=freq_q2, min_val=5.595e9, max_val=5.605e9, unit='Hz 2pi')
tmp1 = c3.c3objs.Quantity(value=anhar_q2, min_val=-380e6, max_val=-120e6, unit='Hz 2pi')
tmp2 = c3.c3objs.Quantity(value=t1_q2, min_val=1e-6, max_val=90e-6, unit='s')
tmp3 = c3.c3objs.Quantity(value=t2star_q2, min_val=10e-6, max_val=90e-6, unit='s')
tmp4 = c3.c3objs.Quantity(value=qubit_temp, min_val=0.0, max_val=0.12, unit='K')
q2 = c3.libraries.chip.Qubit(name="Q2", desc="Qubit 2", freq=tmp0, anhar=tmp1, hilbert_dim=qubit_lvls, t1=tmp2, t2star=tmp3, temp=tmp4)

coupling_strength = 20e6
tmp0 = c3.c3objs.Quantity(value=coupling_strength, min_val=-1 * 1e3, max_val=200e6, unit='Hz 2pi')
q1q2 = c3.libraries.chip.Coupling(name="Q1-Q2", desc="coupling", comment="Coupling qubit 1 to qubit 2",
        connected=["Q1", "Q2"], strength=tmp0, hamiltonian_func=c3.libraries.hamiltonians.int_XX)

drive = c3.libraries.chip.Drive(name="d1", desc="Drive 1", comment="Drive line 1 on qubit 1",
        connected=["Q1"], hamiltonian_func=c3.libraries.hamiltonians.x_drive)
drive2 = c3.libraries.chip.Drive(name="d2", desc="Drive 2", comment="Drive line 2 on qubit 2",
        connected=["Q2"], hamiltonian_func=c3.libraries.hamiltonians.x_drive)


m00_q1 = 0.97  # Prop to read qubit 1 state 0 as 0
m01_q1 = 0.04  # Prop to read qubit 1 state 0 as 1
m00_q2 = 0.96  # Prop to read qubit 2 state 0 as 0
m01_q2 = 0.05  # Prop to read qubit 2 state 0 as 1
one_zeros = np.array([0] * qubit_lvls)
zero_ones = np.array([1] * qubit_lvls)
one_zeros[0] = 1
zero_ones[0] = 0
val1 = one_zeros * m00_q1 + zero_ones * m01_q1
val2 = one_zeros * m00_q2 + zero_ones * m01_q2
min_val = one_zeros * 0.8 + zero_ones * 0.0
max_val = one_zeros * 1.0 + zero_ones * 0.2
confusion_row1 = c3.c3objs.Quantity(value=val1, min_val=min_val, max_val=max_val, unit="")
confusion_row2 = c3.c3objs.Quantity(value=val2, min_val=min_val, max_val=max_val, unit="")
conf_matrix = c3.libraries.tasks.ConfusionMatrix(Q1=confusion_row1, Q2=confusion_row2)

init_temp = 50e-3
tmp0 = c3.c3objs.Quantity(value=init_temp, min_val=-0.001, max_val=0.22, unit='K')
init_ground = c3.libraries.tasks.InitialiseGround(init_temp=tmp0)

# Individual, self-contained components
# Interactions between components
# [conf_matrix, init_ground] # SPAM processing
model = c3.model.Model([q1, q2],  [drive, drive2, q1q2])
model.set_lindbladian(False)
model.set_dressed(True)

sim_res = 100e9 # Resolution for numerical simulation
awg_res = 2e9 # Realistic, limited resolution of an AWG
lo = c3.generator.devices.LO(name='lo', resolution=sim_res)
awg = c3.generator.devices.AWG(name='awg', resolution=awg_res)
mixer = c3.generator.devices.Mixer(name='mixer')
dig_to_an = c3.generator.devices.DigitalToAnalog(name="dac", resolution=sim_res)

tmp0 = c3.c3objs.Quantity(value=1e9, min_val=0.9e9, max_val=1.1e9, unit='Hz/V')
tmp1 = {
    "LO": c3.generator.devices.LO(name='lo', resolution=sim_res, outputs=1),
    "AWG": c3.generator.devices.AWG(name='awg', resolution=awg_res, outputs=1),
    "DigitalToAnalog": c3.generator.devices.DigitalToAnalog(name="dac", resolution=sim_res, inputs=1, outputs=1),
    "Mixer": c3.generator.devices.Mixer(name='mixer', inputs=2, outputs=1),
    "VoltsToHertz": c3.generator.devices.VoltsToHertz(name='v_to_hz', V_to_Hz=tmp0, inputs=1, outputs=1)
}
tmp2 = {
    "d1": {"LO": [], "AWG": [], "DigitalToAnalog": ["AWG"], "Mixer": ["LO", "DigitalToAnalog"], "VoltsToHertz": ["Mixer"]},
    "d2": {"LO": [], "AWG": [], "DigitalToAnalog": ["AWG"], "Mixer": ["LO", "DigitalToAnalog"], "VoltsToHertz": ["Mixer"]},
}
generator = c3.generator.generator.Generator(devices=tmp1, chains=tmp2)
# generator.callback = lambda chain_id, device_id, signal: None #add some callback function here

t_final = 7e-9   # Time for single qubit gates
sideband = 50e6
tmp0 = c3.c3objs.Quantity(value=0.5, min_val=0.2, max_val=0.6, unit="V")
tmp1 = c3.c3objs.Quantity(value=t_final, min_val=0.5*t_final, max_val=1.5*t_final, unit="s")
tmp2 = c3.c3objs.Quantity(value=t_final/4, min_val=t_final/8, max_val=t_final/2, unit="s")
tmp3 = c3.c3objs.Quantity(value=0.0, min_val=-0.5*np.pi, max_val=2.5*np.pi, unit='rad')
tmp4 = c3.c3objs.Quantity(value=-sideband-3e6, min_val=-56*1e6, max_val=-52*1e6 , unit='Hz 2pi')
tmp5 = c3.c3objs.Quantity(value=-1, min_val=-5, max_val=3, unit="")
gauss_params_single = {
    'amp': tmp0,
    't_final': tmp1,
    'sigma': tmp2,
    'xy_angle': tmp3,
    'freq_offset': tmp4,
    'delta': tmp5
}
gauss_env_single = c3.signal.pulse.EnvelopeDrag(name="gauss", desc="Gaussian comp for single-qubit gates",
        params=gauss_params_single, shape=c3.libraries.envelopes.gaussian_nonorm)


tmp0 = c3.c3objs.Quantity(value=t_final, min_val=0.5*t_final, max_val=1.5*t_final, unit="s")
nodrive_env = c3.signal.pulse.Envelope(name="no_drive", params={'t_final':tmp0}, shape=c3.libraries.envelopes.no_drive)


lo_freq_q1 = 5e9 + sideband
tmp0 = c3.c3objs.Quantity(value=lo_freq_q1, min_val=4.5e9, max_val=6e9, unit='Hz 2pi')
tmp1 = c3.c3objs.Quantity(value=0.0, min_val=-np.pi, max_val=3*np.pi, unit='rad')
carrier_parameters = {'freq': tmp0, 'framechange': tmp1}
carr = c3.signal.pulse.Carrier(name="carrier", desc="Frequency of the local oscillator", params=carrier_parameters)


lo_freq_q2 = 5.6e9  + sideband
carr_2 = copy.deepcopy(carr)
carr_2.params['freq'].set_value(lo_freq_q2)
rx90p_q1 = c3.signal.gates.Instruction(name="rx90p", targets=[0], t_start=0.0, t_end=t_final, channels=["d1", "d2"])
rx90p_q2 = c3.signal.gates.Instruction(name="rx90p", targets=[1], t_start=0.0, t_end=t_final, channels=["d1", "d2"])
rx90p_q1.add_component(gauss_env_single, "d1")
rx90p_q1.add_component(carr, "d1")
rx90p_q2.add_component(copy.deepcopy(gauss_env_single), "d2")
rx90p_q2.add_component(carr_2, "d2")

rx90p_q1.add_component(nodrive_env, "d2")
rx90p_q1.add_component(copy.deepcopy(carr_2), "d2")
rx90p_q1.comps["d2"]["carrier"].params["framechange"].set_value((-sideband * t_final) * 2 * np.pi % (2 * np.pi))
rx90p_q2.add_component(nodrive_env, "d1")
rx90p_q2.add_component(copy.deepcopy(carr), "d1")
rx90p_q2.comps["d1"]["carrier"].params["framechange"].set_value((-sideband * t_final) * 2 * np.pi % (2 * np.pi))

ry90p_q1 = copy.deepcopy(rx90p_q1)
ry90p_q1.name = "ry90p"
rx90m_q1 = copy.deepcopy(rx90p_q1)
rx90m_q1.name = "rx90m"
ry90m_q1 = copy.deepcopy(rx90p_q1)
ry90m_q1.name = "ry90m"
ry90p_q1.comps['d1']['gauss'].params['xy_angle'].set_value(0.5 * np.pi)
rx90m_q1.comps['d1']['gauss'].params['xy_angle'].set_value(np.pi)
ry90m_q1.comps['d1']['gauss'].params['xy_angle'].set_value(1.5 * np.pi)
single_q_gates = [rx90p_q1, ry90p_q1, rx90m_q1, ry90m_q1]

ry90p_q2 = copy.deepcopy(rx90p_q2)
ry90p_q2.name = "ry90p"
rx90m_q2 = copy.deepcopy(rx90p_q2)
rx90m_q2.name = "rx90m"
ry90m_q2 = copy.deepcopy(rx90p_q2)
ry90m_q2.name = "ry90m"
ry90p_q2.comps['d2']['gauss'].params['xy_angle'].set_value(0.5 * np.pi)
rx90m_q2.comps['d2']['gauss'].params['xy_angle'].set_value(np.pi)
ry90m_q2.comps['d2']['gauss'].params['xy_angle'].set_value(1.5 * np.pi)
single_q_gates.extend([rx90p_q2, ry90p_q2, rx90m_q2, ry90m_q2])


parameter_map = c3.parametermap.ParameterMap(instructions=single_q_gates, model=model, generator=generator)

exp = c3.experiment.Experiment(pmap=parameter_map)
exp.set_opt_gates(['rx90p[0]'])
unitaries = exp.compute_propagators()
tmp0 = np.asarray(unitaries['rx90p[0]']) #(np,complex128,(9,9))

psi_init = [[0] * 9]
psi_init[0][0] = 1
init_state = tf.transpose(tf.constant(psi_init, tf.complex128))

barely_a_seq = ['rx90p[0]']


def plot_dynamics(exp, psi_init, seq):
    """
    Plotting code for time-resolved populations.

    psi_init: tf.Tensor
        Initial state or density matrix.
    seq: list
        List of operations to apply to the initial state.
    """
    model = exp.pmap.model
    exp.compute_propagators()
    dUs = exp.partial_propagators
    psi_t = psi_init.numpy()
    pop_t = exp.populations(psi_t, model.lindbladian)
    for gate in seq:
        for du in dUs[gate]:
            psi_t = np.matmul(du.numpy(), psi_t)
            pops = exp.populations(psi_t, model.lindbladian)
            pop_t = np.append(pop_t, pops, axis=1)

    fig, axs = plt.subplots(1, 1)
    ts = exp.ts
    dt = ts[1] - ts[0]
    ts = np.linspace(0.0, dt*pop_t.shape[1], pop_t.shape[1])
    axs.plot(ts / 1e-9, pop_t.T)
    axs.grid(linestyle="--")
    axs.tick_params(direction="in", left=True, right=True, top=True, bottom=True)
    axs.set_xlabel('Time [ns]')
    axs.set_ylabel('Population')
    plt.legend(model.state_labels)
    fig.savefig('tbd00.png')

plot_dynamics(exp, init_state, barely_a_seq)

plot_dynamics(exp, init_state, barely_a_seq * 5)

qc = qiskit.QuantumCircuit(2)
qc.append(c3.qiskit.c3_gates.RX90pGate(), [0])
# qc.draw()

c3_provider = c3.qiskit.C3Provider()
c3_backend = c3_provider.get_backend("c3_qasm_physics_simulator")
c3_backend.set_c3_experiment(exp)

c3_job_unopt = c3_backend.run(qc)
result_unopt = c3_job_unopt.result()
res_pops_unopt = result_unopt.data()["state_pops"]
# qiskit.tools.visualization.plot_histogram(res_pops_unopt, title='Simulation of Qiskit circuit with Unoptimized Gates')

# open-loop optimal control
opt_gates = ["rx90p[0]"]
gateset_opt_map=[
    [("rx90p[0]", "d1", "gauss", "amp")],
    [("rx90p[0]", "d1", "gauss", "freq_offset")],
    [("rx90p[0]", "d1", "gauss", "xy_angle")],
    [("rx90p[0]", "d1", "gauss", "delta")],
    [("rx90p[0]", "d1", "carrier", "framechange")]
]
parameter_map.set_opt_map(gateset_opt_map)
parameter_map.print_parameters()

log_dir = hf_logdir('c3logs')
opt = c3.optimizers.optimalcontrol.OptimalControl(
    dir_path=log_dir,
    fid_func=c3.libraries.fidelities.unitary_infid_set,
    fid_subspace=["Q1", "Q2"],
    pmap=parameter_map,
    algorithm=c3.libraries.algorithms.lbfgs,
    options={"maxfun" : 150},
    run_name="better_X90"
)
exp.set_opt_gates(opt_gates)
opt.set_exp(exp)
opt.optimize_controls()
opt.current_best_goal

plot_dynamics(exp, init_state, barely_a_seq)
plot_dynamics(exp, init_state, barely_a_seq * 5)
parameter_map.print_parameters()

c3_job_opt = c3_backend.run(qc)
result_opt = c3_job_opt.result()
res_pops_opt = result_opt.data()["state_pops"]
