# https://docs.q-ctrl.com/boulder-opal/user-guides/how-to-optimize-controls-robust-to-strong-noise-sources
import numpy as np
import matplotlib.pyplot as plt
import dotenv

import qctrl
import qctrlvisualizer

plt.style.use(qctrlvisualizer.get_qctrl_style())

QCTRL_ORGANIZATION = dotenv.dotenv_values('.env')['QCTRL_ORGANIZATION']
QCTRL_HANDLE = qctrl.Qctrl(organization=QCTRL_ORGANIZATION)
QCTRL_HANDLE.request_machines(1)

# Define physical constraints.
duration = 2e-6  # s
gamma_max = 2 * np.pi * 0.5e6  # rad/s
segment_count = 20
batch_size = 200

# Create graph object.
graph = QCTRL_HANDLE.create_graph()

# Define optimizable controls.
gamma_x = graph.utils.real_optimizable_pwc_signal(
    segment_count=segment_count,
    minimum=-gamma_max,
    maximum=gamma_max,
    duration=duration,
    name="gamma_x",
)
gamma_y = graph.utils.real_optimizable_pwc_signal(
    segment_count=segment_count,
    minimum=-gamma_max,
    maximum=gamma_max,
    duration=duration,
    name="gamma_y",
)

# Create noise signals, aᵢ cos(ωᵢt + ϕᵢ).
noise_signals = []
sample_times = (0.5 + np.arange(segment_count)) * duration / segment_count
for _ in range(10):
    a = graph.random_normal(shape=(batch_size, 1), mean=0.0, standard_deviation=0.05)
    omega = graph.random_uniform(
        shape=(batch_size, 1), lower_bound=np.pi, upper_bound=2 * np.pi
    )
    phi = graph.random_uniform(
        shape=(batch_size, 1), lower_bound=0.0, upper_bound=2 * np.pi
    )
    noise_signals.append(
        graph.pwc_signal(
            values=a * graph.cos(omega * sample_times[None] + phi), duration=duration
        )
    )

# Define Hamiltonian.
total_noise = graph.pwc_sum(noise_signals)
hamiltonian = (1 + total_noise) * (
    gamma_x * graph.pauli_matrix("X") + gamma_y * graph.pauli_matrix("Y")
)

# Create infidelity.
infidelities = graph.infidelity_pwc(
    hamiltonian, target=graph.target(graph.pauli_matrix("X")), name="infidelities"
)

# Define cost (average infidelity).
cost = graph.sum(infidelities) / batch_size
cost.name = "cost"

# Run the optimization.
optimization_result = QCTRL_HANDLE.functions.calculate_stochastic_optimization(
    graph=graph,
    cost_node_name="cost",
    output_node_names=["gamma_x", "gamma_y", "infidelities"],
    iteration_count=10000,
    target_cost=1e-6,
)

print(f"\nOptimized cost:\t {optimization_result.best_cost:.3e}\n")

# Plot histogram of infidelities evaluated at the optimized variables.
print(
    f'Batch mean: {optimization_result.best_output["infidelities"]["value"].mean():.2e}, '
    f'standard deviation: {optimization_result.best_output["infidelities"]["value"].std():.2e}'
)

plt.title("Batch of optimized infidelities")
plt.xlabel("Infidelities")
plt.ylabel("Count")
plt.hist(optimization_result.best_output["infidelities"]["value"], bins="auto", ec="black")
plt.xlim([0.0, 1e-4])

# Plot the optimized controls.
qctrlvisualizer.plot_controls(
    {
        r"$\gamma_x$": optimization_result.best_output["gamma_x"],
        r"$\gamma_y$": optimization_result.best_output["gamma_y"],
    }
)
