import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import dotenv

import qctrl
import qctrlvisualizer

# plt.style.use(qctrlvisualizer.get_qctrl_style())

# fail to authenticate on vscode
QCTRL_ORGANIZATION = dotenv.dotenv_values('.env')['QCTRL_ORGANIZATION']
QCTRL_HANDLE = qctrl.Qctrl(organization=QCTRL_ORGANIZATION)
QCTRL_HANDLE.request_machines(1)


# Total gate duration.
gate_duration = 4e-6  # s
sigma = gate_duration / 6

times, dt = np.linspace(0, gate_duration, 64, retstep=True)
tmp0 = np.exp(-((times - gate_duration / 2) ** 2) / 2 / sigma**2)
drive_values = np.pi * tmp0 / np.sum(tmp0 * dt)

# qctrlvisualizer.plot_controls({r"$\gamma$": QCTRL_HANDLE.utils.pwc_arrays_to_pairs(gate_duration, drive_values)})

dim = 4
alpha = 2 * np.pi * 0.45e6  # rad.Hz

transmon = QCTRL_HANDLE.superconducting.Transmon(dimension=dim, anharmonicity=alpha, drive=drive_values)

q0 = np.zeros(dim)
q0[0] = 1
result = QCTRL_HANDLE.superconducting.simulate(transmons=[transmon],
        cavities=[], interactions=[], gate_duration=gate_duration, initial_state=q0)
# result['sample_times'] #(np,float64,128)
# result['unitaries'] #(np,complex128,(128,4,4))
# result['state_evolution'] #(np,complex128,(128,4))
z0 = result['unitaries']
assert np.abs(z0 @ z0.transpose(0,2,1).conj() - np.eye(z0.shape[1])).max() < 1e-10
assert np.abs(z0[0]-np.eye(z0.shape[1])).max() < 1e-10
z1 = z0[1:] @ z0[:-1].transpose(0,2,1).conj()
tmp0 = np.stack([scipy.linalg.logm(x)/1j for x in z1])
matH = tmp0 - np.trace(tmp0, axis1=1, axis2=2).real.reshape(-1,1,1)*np.eye(tmp0.shape[1])/tmp0.shape[1]
assert np.abs(matH - matH.transpose(0,2,1).conj()).max() < 1e-10
tmp0 = matH.reshape(matH.shape[0], -1)
tmp1 = np.concatenate([tmp0.real, tmp0.imag], axis=1)
EVL = np.linalg.eigvalsh(tmp1 @ tmp1.T)
assert np.abs(result['unitaries'] @ q0 - result['state_evolution']).max() < 1e-10

probability = np.abs(result["state_evolution"]) ** 2
# tmp0 = {rf"$|{idx}\rangle$": probability[:, idx] for idx in range(dim)}
# qctrlvisualizer.plot_population_dynamics(result["sample_times"], tmp0)


unitary_target = np.zeros((dim,dim), dtype=np.float64)
unitary_target[[0, 1], [1, 0]] = 1.0


## optimization
# Gate duration and number of optimizable piecewise-constant segments.
gate_duration = 5e-6  # s
segment_count = 20

# Physical properties of the transmon.
transmon_dimension = 5
alpha = 2 * np.pi * 0.3e6  # rad.Hz
gamma_max = 2 * np.pi * 0.3e6  # rad.Hz

# Create transmon object with optimizable drive.
transmon = QCTRL_HANDLE.superconducting.Transmon(
    dimension=transmon_dimension,
    anharmonicity=alpha,
    drive=QCTRL_HANDLE.superconducting.ComplexOptimizableSignal(segment_count, 0, gamma_max),
)

# Physical properties of the cavity.
cavity_dimension = 4
K = 2 * np.pi * 4e5  # rad.Hz
delta = -2 * np.pi * 0.3e6  # rad.Hz

# Create cavity object.
cavity = QCTRL_HANDLE.superconducting.Cavity(
    dimension=cavity_dimension, frequency=delta, kerr_coefficient=K
)

# Physical properties of the interaction.
omega_max = 2 * np.pi * 0.2e6  # rad.Hz

# Create interaction object.
interaction = QCTRL_HANDLE.superconducting.TransmonCavityInteraction(
    rabi_coupling=QCTRL_HANDLE.superconducting.ComplexOptimizableSignal(
        segment_count, 0, omega_max
    )
)


# Define initial state |00>.
transmon_ground_state = np.zeros(transmon_dimension)
transmon_ground_state[0] = 1
cavity_ground_state = np.zeros(cavity_dimension)
cavity_ground_state[0] = 1
initial_state = np.kron([transmon_ground_state], [cavity_ground_state])[0]

# Define target state (|00> + i |11>) / √2.
transmon_excited_state = np.zeros(transmon_dimension)
transmon_excited_state[1] = 1
cavity_excited_state = np.zeros(cavity_dimension)
cavity_excited_state[1] = 1
excited_state = np.kron([transmon_excited_state], [cavity_excited_state])[0]

target_state = (initial_state + 1j * excited_state) / np.sqrt(2)

optimization_result = QCTRL_HANDLE.superconducting.optimize(
    transmons=[transmon],
    cavities=[cavity],
    interactions=[interaction],
    gate_duration=gate_duration,
    initial_state=initial_state,
    target_state=target_state,
)
# optimization_result['optimized_values']['transmon.drive'] #(np,complex128,20)
# optimization_result['optimized_values']['transmon_cavity_interaction.rabi_coupling'] #(np,complex128,20)
# optimization_result['sample_times'] #(np,float64,(128))
# optimization_result['unitaries'] #(np,complex128,(128,20,20))
# optimization_result['infidelity'] #float
# optimization_result['state_evolution'] #(np,complex128,(128,20))
infidelity = 1-abs(np.vdot(optimization_result['state_evolution'][-1], target_state))**2
