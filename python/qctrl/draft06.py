# https://docs.q-ctrl.com/boulder-opal/tutorials/simulate-the-dynamics-of-a-single-qubit-using-computational-graphs
import numpy as np
import matplotlib.pyplot as plt
import dotenv

import qctrl
import qctrlvisualizer

plt.style.use(qctrlvisualizer.get_qctrl_style())

QCTRL_ORGANIZATION = dotenv.dotenv_values('.env')['QCTRL_ORGANIZATION']
QCTRL_HANDLE = qctrl.Qctrl(organization=QCTRL_ORGANIZATION)
QCTRL_HANDLE.request_machines(1)


graph = QCTRL_HANDLE.create_graph()

# (Constant) Hamiltonian parameters.
omega = 2 * np.pi * 1.0e6  # rad/s
delta = 2 * np.pi * 0.4e6  # rad/s

# Total duration of the simulation.
duration = 2e-6  # s

# Hamiltonian term coefficients.
omega_signal = graph.constant_pwc(constant=omega, duration=duration)
delta_signal = graph.constant_pwc(constant=delta, duration=duration)

# Total Hamiltonian, [Ω σ- + H.c.]/2 + δ σz.
hamiltonian = graph.hermitian_part(omega_signal * graph.pauli_matrix("M")) + delta_signal * graph.pauli_matrix("Z")

# Times at which to sample the simulation.
sample_times = np.linspace(0.0, duration, 100)

# Time-evolution operators, U(t).
unitaries = graph.time_evolution_operators_pwc(hamiltonian=hamiltonian, sample_times=sample_times, name="unitaries")

# Initial state of the qubit, |0⟩.
initial_state = graph.fock_state(2, 0)[:, None]

# Evolved states, |ψ(t)⟩ = U(t) |0⟩
evolved_states = unitaries @ initial_state
evolved_states.name = "states"

result = QCTRL_HANDLE.functions.calculate_graph(graph=graph, output_node_names=["unitaries", "states"])

unitaries = result.output["unitaries"]["value"]
print(f"Shape of calculated unitaries: {unitaries.shape}")

states = result.output["states"]["value"]
print(f"Shape of calculated evolved states: {states.shape}")

# Calculate qubit populations |⟨ψ|0⟩|².
qubit_populations = np.abs(states.squeeze()) ** 2
# Plot populations.
qctrlvisualizer.plot_population_dynamics(sample_times, {rf"$|{k}\rangle$": qubit_populations[:, k] for k in [0, 1]})
# qctrlvisualizer.display_bloch_sphere(states.squeeze())


# Gaussian pulse parameters.
omega_max = 2.0 * np.pi * 1e6  # Hz
segment_count = 50
times = np.linspace(-3, 3, segment_count)
omega_values = -1j * omega_max * np.exp(-(times**2))

# Total duration of the pulse to achieve a π/2 gate.
pulse_duration = 0.5 * segment_count * np.pi / np.sum(np.abs(omega_values))

# Plot Gaussian pulse.
qctrlvisualizer.plot_controls({"$\\Omega$": QCTRL_HANDLE.utils.pwc_arrays_to_pairs(pulse_duration, omega_values)}, polar=False)

graph = QCTRL_HANDLE.create_graph()

# Times at which to sample the simulation.
sample_times = np.linspace(0.0, pulse_duration, 100)

# Time-dependent Hamiltonian term coefficient.
omega_signal = graph.pwc_signal(values=omega_values, duration=pulse_duration)

# Total Hamiltonian, [Ω σ- + H.c.]/2
hamiltonian = graph.hermitian_part(omega_signal * graph.pauli_matrix("M"))

# Time-evolution operators, U(t).
unitaries = graph.time_evolution_operators_pwc(hamiltonian=hamiltonian, sample_times=sample_times, name="unitaries")

# Initial state of the qubit, |0⟩.
initial_state = graph.fock_state(2, 0)[:, None]

# Evolved states, |ψ(t)⟩ = U(t) |0⟩
evolved_states = unitaries @ initial_state
evolved_states.name = "states"

# Execute the graph.
result = QCTRL_HANDLE.functions.calculate_graph(graph=graph, output_node_names=["unitaries", "states"])

# Retrieve values of the calculation
unitaries = result.output["unitaries"]["value"]
states = result.output["states"]["value"]

print("Unitary gate implemented by the Gaussian pulse:")
print(unitaries[-1])
print("Final state after the gate:")
print(states[-1])

# Calculate qubit populations |⟨ψ|0⟩|².
qubit_populations = np.abs(states.squeeze()) ** 2
qctrlvisualizer.plot_population_dynamics(sample_times, {rf"$|{k}\rangle$": qubit_populations[:, k] for k in [0, 1]})
