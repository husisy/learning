# https://docs.q-ctrl.com/boulder-opal/user-guides/how-to-automate-closed-loop-hardware-optimization
# https://docs.q-ctrl.com/boulder-opal/user-guides/how-to-optimize-controls-starting-from-an-incomplete-system-model
import numpy as np
import matplotlib.pyplot as plt
import dotenv

import qctrl
import qctrlvisualizer

plt.style.use(qctrlvisualizer.get_qctrl_style())

QCTRL_ORGANIZATION = dotenv.dotenv_values('.env')['QCTRL_ORGANIZATION']
QCTRL_HANDLE = qctrl.Qctrl(organization=QCTRL_ORGANIZATION)
QCTRL_HANDLE.request_machines(1)


# Define standard deviation of the errors in the experimental results.
sigma = 0.01

# Define standard matrices.
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)

# Define control parameters.
duration = 1e-6  # s

# Create a random unknown operator.
rng = np.random.default_rng(seed=10)
phi = rng.uniform(-np.pi, np.pi)
u = rng.uniform(-1, 1)
Q_unknown = (u * sigma_z + np.sqrt(1 - u**2) * (np.cos(phi) * sigma_x + np.sin(phi) * sigma_y)) / 4


z1 = []

# Define simulation of a quantum experiment to use in the optimization loop.
def run_experiments(omegas):
    """
    Simulates a series of experiments where controls `omegas` attempt to apply
    an X gate to a system. The result of each experiment is the infidelity plus
    a Gaussian error.

    In your actual implementation, this function would run the experiment with
    the parameters passed. Note that the simulation handles multiple test points,
    while your experimental implementation might need to queue the test point
    requests to obtain one at a time from the apparatus.
    """

    # Create the graph with the dynamics of the system.
    graph = QCTRL_HANDLE.create_graph()
    signal = graph.pwc_signal(values=omegas, duration=duration)
    graph.infidelity_pwc(
        hamiltonian=0.5 * signal * (sigma_x + Q_unknown),
        target=graph.target(operator=sigma_x),
        name="infidelities",
    )

    # Run the simulation.
    result = QCTRL_HANDLE.functions.calculate_graph(
        graph=graph, output_node_names=["infidelities"]
    )

    # Add error to the measurement.
    error_values = rng.normal(loc=0, scale=sigma, size=len(omegas))
    infidelities = result.output["infidelities"]["value"] + error_values
    z1.append(error_values.copy())

    # Return only infidelities between 0 and 1.
    return np.clip(infidelities, 0, 1)

# Define the number of test points obtained per run.
test_point_count = 20

# Define number of segments in the control.
segment_count = 10

# Define initial parameter set as constant controls with piecewise constant segments.
initial_test_parameters = (np.pi / duration) * (np.linspace(-1, 1, test_point_count)[:, None]) * np.ones(segment_count)


# Define length-scale bounds as a NumPy array.
length_scale_bounds = np.repeat([[1e-5, 1e5]], segment_count, axis=0)

# Create Gaussian Process optimizer object.
gp_optimizer = QCTRL_HANDLE.closed_loop.GaussianProcess(
    length_scale_bounds=length_scale_bounds, seed=0
)

# Define bounds for all optimization parameters.
bounds = np.repeat([[-1, 1]], segment_count, axis=0) * 5 * np.pi / duration

# Create random number generator for the simulation function.
simulation_seed = 5
rng = np.random.default_rng(simulation_seed)

# Run the closed-loop optimization.
gp_results = QCTRL_HANDLE.closed_loop.optimize(
    cost_function=run_experiments,
    initial_test_parameters=initial_test_parameters,
    optimizer=gp_optimizer,
    bounds=bounds,
    target_cost=3 * sigma,
    cost_uncertainty=sigma,
)

# Print optimized cost and plot optimized controls.
print(f"\nOptimized infidelity: {gp_results['best_cost']:.5f}")
qctrlvisualizer.plot_controls(
    {
        r"$\Omega(t)$": QCTRL_HANDLE.utils.pwc_arrays_to_pairs(
            duration, gp_results["best_parameters"]
        )
    }
)

import scipy.linalg
z0 = np.eye(2)
for ind0 in range(segment_count):
    tmp0 = 0.5*gp_results['best_parameters'][ind0]*(sigma_x+Q_unknown)
    z0 = scipy.linalg.expm(tmp0 * (-1j*duration/segment_count)) @ z0
z2 = 1-abs(np.trace(z0.T.conj() @ sigma_x))**2/(sigma_x.shape[0]**2)
np.abs((z2 + z1[-1]).min() - gp_results['best_cost']).min()
