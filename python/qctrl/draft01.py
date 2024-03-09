# https://docs.q-ctrl.com/boulder-opal/user-guides/how-to-perform-parameter-estimation-with-a-small-amount-of-data
# https://docs.q-ctrl.com/boulder-opal/tutorials/estimate-parameters-of-a-single-qubit-hamiltonian
import numpy as np
import matplotlib.pyplot as plt
import dotenv

import qctrl
import qctrlvisualizer

plt.style.use(qctrlvisualizer.get_qctrl_style())

QCTRL_ORGANIZATION = dotenv.dotenv_values('.env')['QCTRL_ORGANIZATION']
QCTRL_HANDLE = qctrl.Qctrl(organization=QCTRL_ORGANIZATION)
QCTRL_HANDLE.request_machines(1)


actual_Omegas = (2 * np.pi) * np.array([0.5e6, 1.5e6, 1.8e6])  # Hz

def run_experiments(wait_times, initial_states, projector_states):
    """
    Runs a batch of simulated experiments for the given set of `wait_times`,
    `initial_states`, and `projector_states`, whose population is measured at
    the end.
    """
    graph = QCTRL_HANDLE.create_graph()
    tmp0 = (actual_Omegas[0] * graph.pauli_matrix("X") + actual_Omegas[1] * graph.pauli_matrix("Y") + actual_Omegas[2] * graph.pauli_matrix("Z"))/ 2
    hamiltonian = graph.constant_pwc(constant=tmp0, duration=max(wait_times))
    unitaries = graph.time_evolution_operators_pwc(hamiltonian=hamiltonian, sample_times=wait_times)
    tmp0 = graph.abs(graph.adjoint(projector_states)[:, None, ...] @ unitaries @ initial_states[:, None, ...])**2
    populations = graph.reshape(tmp0, (len(initial_states), len(wait_times)))

    # Add Gaussian error to the measurement result.
    measurement_error = graph.random_normal(shape=populations.shape, mean=0, standard_deviation=0.01)
    measurement_results = graph.add(populations, measurement_error, name="measurement_results")
    result = QCTRL_HANDLE.functions.calculate_graph(graph=graph, output_node_names=["measurement_results"])
    return result.output["measurement_results"]["value"]


# Range of wait times in the experiments.
duration = 500e-9  # s
time_segment_count = 20
time_resolution = duration / time_segment_count
wait_times = np.linspace(time_resolution, duration, time_segment_count)

ket_xp = np.array([[1], [1]]) / np.sqrt(2)
ket_yp = np.array([[1], [1j]]) / np.sqrt(2)
ket_zp = np.array([[1], [0]])
initial_states = np.array([ket_zp, ket_xp, ket_yp])
projector_states = np.array([ket_yp, ket_zp, ket_xp])

measurement_results = run_experiments(wait_times, initial_states, projector_states)


graph = QCTRL_HANDLE.create_graph()

# Parameters to be estimated. Frequencies whose half-periods are shorter than the smaller spacing between points are out of bounds.
frequency_bound = 1 / time_resolution
Omegas = graph.optimization_variable(3, lower_bound=-frequency_bound, upper_bound=frequency_bound, name="Omegas")

tmp0 = (Omegas[0] * graph.pauli_matrix("X") + Omegas[1] * graph.pauli_matrix("Y") + Omegas[2] * graph.pauli_matrix("Z"))/ 2
hamiltonian = graph.constant_pwc(constant=tmp0, duration=duration)
unitaries = graph.time_evolution_operators_pwc(hamiltonian=hamiltonian, sample_times=wait_times)
tmp0 = graph.abs(graph.adjoint(projector_states[:, None, ...]) @ unitaries @ initial_states[:, None, ...])**2
populations = graph.reshape(tmp0, (len(initial_states), len(wait_times)))
cost = graph.sum((populations - measurement_results) ** 2, name="rss")
hessian_matrix = graph.hessian(cost, [Omegas], name="hessian")


result = QCTRL_HANDLE.functions.calculate_optimization(graph=graph, cost_node_name="rss", output_node_names=["Omegas", "hessian"], optimization_count=20)
estimated_Omegas = result.output["Omegas"]["value"]
hessian = result.output["hessian"]["value"]
cost_rss = result.cost

# Plot 95%-confidence ellipses.
measurement_count = np.prod(measurement_results.shape)
confidence_region = QCTRL_HANDLE.utils.confidence_ellipse_matrix(hessian, cost_rss, measurement_count)

Omega_names = [r"$\Omega_x$", r"$\Omega_y$", r"$\Omega_z$"]
qctrlvisualizer.plot_confidence_ellipses(confidence_region, estimated_Omegas, actual_Omegas, Omega_names)
plt.suptitle("Estimated Hamiltonian parameters", y=1.05)


import scipy.linalg
PauliX = np.array([[0, 1], [1, 0]])
PauliY = np.array([[0, -1j], [1j, 0]])
PauliZ = np.array([[1, 0], [0, -1]])

tmp0 = np.array([1,1]) / np.sqrt(2)
tmp1 = np.array([1,1j]) / np.sqrt(2)
tmp2 = np.array([1,0])
initial_state = np.array([tmp2, tmp0, tmp1])
measure_op = np.stack([x.reshape(-1,1)*x.conj() for x in [tmp1,tmp2,tmp0]], axis=0) #TODO conj ?
# projector_states = np.array([tmp1, tmp2, tmp0])

theta = result.output["Omegas"]["value"]
# tspan = np.linspace(0, 500e-9, 20)
duration = 500e-9  # s

num_measure = 20
t_measure_list = np.linspace(1/num_measure, 1, num_measure)*duration


hamiltonian = (theta[0] * PauliX + theta[1] * PauliY + theta[2] * PauliZ)/ 2
tmp0 = np.einsum(scipy.linalg.expm((-1j*t_measure_list).reshape(-1,1,1)*hamiltonian), [0,1,2], initial_state, [3,2], [3,0,1], optimize=True)
tmp1 = np.einsum(measure_op, [0,1,2], tmp0, [0,3,2], tmp0.conj(), [0,3,1], [0,3], optimize=True).real
loss = np.sum((tmp1 - measurement_results)**2)
assert np.abs(loss - result.cost) < 1e-10
