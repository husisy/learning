# https://docs.q-ctrl.com/boulder-opal/tutorials/design-robust-single-qubit-gates-using-computational-graphs
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


# Pulse parameters.
segment_count = 50
duration = 10e-6  # s

# Maximum value for |α(t)|.
alpha_max = 2 * np.pi * 0.25e6  # rad/s

# Real PWC signal representing α(t).
alpha = graph.utils.real_optimizable_pwc_signal(
    segment_count=segment_count,
    duration=duration,
    minimum=-alpha_max,
    maximum=alpha_max,
    name="$\\alpha$",
)

# Maximum value for |γ(t)|.
gamma_max = 2 * np.pi * 0.5e6  # rad/s

# Complex PWC signal representing γ(t)
gamma = graph.utils.complex_optimizable_pwc_signal(
    segment_count=segment_count, duration=duration, maximum=gamma_max, name="$\\gamma$"
)

# Detuning δ.
delta = 2 * np.pi * 0.25e6  # rad/s

# Total Hamiltonian.
hamiltonian = (
    alpha * graph.pauli_matrix("Z")
    + graph.hermitian_part(gamma * graph.pauli_matrix("M"))
    + delta * graph.pauli_matrix("Z")
)

# Target operation node.
target = graph.target(operator=graph.pauli_matrix("Y"))

# Dephasing noise amplitude.
beta = 2 * np.pi * 20e3  # rad/s

# (Constant) dephasing noise term.
dephasing = beta * graph.pauli_matrix("Z")

# Robust infidelity.
robust_infidelity = graph.infidelity_pwc(
    hamiltonian=hamiltonian,
    noise_operators=[dephasing],
    target=target,
    name="robust_infidelity",
)

optimization_result = QCTRL_HANDLE.functions.calculate_optimization(
    graph=graph,
    cost_node_name="robust_infidelity",
    output_node_names=["$\\alpha$", "$\\gamma$"],
)

print(f"Optimized robust cost: {optimization_result.cost:.3e}")

qctrlvisualizer.plot_controls(controls=optimization_result.output)



# Retrieve values of the robust PWC controls α(t) and γ(t).
_, alpha_values, _ = QCTRL_HANDLE.utils.pwc_pairs_to_arrays(
    optimization_result.output["$\\alpha$"]
)
_, gamma_values, _ = QCTRL_HANDLE.utils.pwc_pairs_to_arrays(
    optimization_result.output["$\\gamma$"]
)

# Create a new Boulder Opal graph.
graph = QCTRL_HANDLE.create_graph()

# Create a real PWC signal representing α(t).
alpha = graph.pwc_signal(values=alpha_values, duration=duration)

# Create a complex PWC signal representing γ(t).
gamma = graph.pwc_signal(values=gamma_values, duration=duration)

# Values of β to scan over.
beta_scan = np.linspace(-beta, beta, 100)

# 1D batch of constant PWC
dephasing_amplitude = graph.constant_pwc(
    beta_scan, duration=duration, batch_dimension_count=1
)

# Total Hamiltonian.
hamiltonian = (
    alpha * graph.pauli_matrix("Z")
    + graph.hermitian_part(gamma * graph.pauli_matrix("M"))
    + delta * graph.pauli_matrix("Z")
    + dephasing_amplitude * graph.pauli_matrix("Z")
)

# Target operation node.
target = graph.target(operator=graph.pauli_matrix("Y"))

# Quasi-static scan infidelity.
infidelity = graph.infidelity_pwc(
    hamiltonian=hamiltonian, target=target, name="infidelity"
)

quasi_static_scan_result = QCTRL_HANDLE.functions.calculate_graph(
    graph=graph, output_node_names=["infidelity"]
)

# Array with the scanned infidelities.
infidelities = quasi_static_scan_result.output["infidelity"]["value"]

# Create plot with the infidelity scan.
fig, ax = plt.subplots()
ax.plot(beta_scan / 1e6 / 2 / np.pi, infidelities)
ax.set_xlabel(r"$\beta$ (MHz)")
ax.set_ylabel("Infidelity")
