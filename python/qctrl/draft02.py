# https://docs.q-ctrl.com/boulder-opal/user-guides/how-to-perform-parameter-estimation-with-a-large-amount-of-data
import numpy as np
import dotenv

import qctrl

QCTRL_ORGANIZATION = dotenv.dotenv_values('.env')['QCTRL_ORGANIZATION']
QCTRL_HANDLE = qctrl.Qctrl(organization=QCTRL_ORGANIZATION)
QCTRL_HANDLE.request_machines(1)


# Parameters to estimate
actual_alpha = 2 * np.pi * 0.83e6  # Hz
actual_delta = 2 * np.pi * 0.31e6  # Hz
actual_gamma = 2 * np.pi * 0.11e6  # Hz

num_qubit = 5
num_shot = 100
dataset_size = 5000
batch_size = 100

# Pulses parameters
duration = 10.0e-6  # s
segment_count = 10

# Generate values for all Rabi couplings pulses in the dataset, shape [D=5000,T=10]
omega_dataset = np.random.uniform(low=0.0, high=1.0, size=[dataset_size, segment_count])


graph = QCTRL_HANDLE.create_graph()
initial_state = graph.fock_state(2**num_qubit, 0) #|00000>
omega_signal = graph.pwc_signal(values=omega_dataset, duration=duration)

# Create Hamiltonian, batch of D [N,N] pwc operators
# X1 + X2 + ... + Xn
tmp0 = sum(graph.pauli_kronecker_product([("X", k)], num_qubit) for k in range(num_qubit))
# Z1 + Z2 + ... + Zn
tmp1 = sum(graph.pauli_kronecker_product([("Z", k)], num_qubit) for k in range(num_qubit))
# Z1 * Z2 * ... * Zn
tmp2 = graph.pauli_kronecker_product([("Z", i) for i in range(num_qubit)], num_qubit)
hamiltonian = omega_signal * actual_alpha * tmp0 + actual_delta * tmp1 + actual_gamma * tmp2

# Calculate final unitary evolution operator, shape=[D,N,N]
unitary = graph.time_evolution_operators_pwc(hamiltonian=hamiltonian, sample_times=np.array([duration]))[:, -1]

# Evolve intial state, shape=[D,N,1]
final_state = unitary @ initial_state[:, None]
populations = graph.abs(final_state[:, :, 0]) ** 2
populations.name = "populations"
result = QCTRL_HANDLE.functions.calculate_graph(graph=graph, output_node_names=["populations"])
calculated_populations = result.output["populations"]["value"] #probablity (np,float64,(5000,32)))

# Take num_shot projective measurements of the system.
# binomial distribution
measurement_dataset = np.array([(np.random.choice(2**num_qubit, size=num_shot, p=x)==0).mean() for x in calculated_populations])


graph = QCTRL_HANDLE.create_graph()
initial_state = graph.fock_state(2**num_qubit, 0)

# Sample a batch of the omega/measurement dataset
omega_values_batch, measurement_batch = graph.random_choices(data=[omega_dataset, measurement_dataset], sample_count=batch_size)

# Parameters to be estimated
alpha = graph.optimization_variable(count=1, lower_bound=0, upper_bound=10.0e6)[0]
alpha.name = "alpha"
delta = graph.optimization_variable(count=1, lower_bound=0, upper_bound=5.0e6)[0]
delta.name = "delta"
gamma = graph.optimization_variable(count=1, lower_bound=0, upper_bound=2.0e6)[0]
gamma.name = "gamma"

# Create signal, batch of D pwc signals
omega_signal = graph.pwc_signal(values=omega_values_batch, duration=duration)

# Create Hamiltonian, batch of D N×N pwc operators
tmp0 = sum(graph.pauli_kronecker_product([("X", k)], num_qubit) for k in range(num_qubit))
tmp1 = sum(graph.pauli_kronecker_product([("Z", k)], num_qubit) for k in range(num_qubit))
tmp2 = graph.pauli_kronecker_product([("Z", i) for i in range(num_qubit)], num_qubit)
hamiltonian = omega_signal * alpha * tmp0 + delta * tmp1 + gamma * tmp2

# Calculate final unitary evolution operator, shape=[D,N,N]
unitary = graph.time_evolution_operators_pwc(hamiltonian=hamiltonian, sample_times=np.array([duration]))[:, -1]
state = unitary @ initial_state[:, None]
populations = graph.abs(state[:, :, 0]) ** 2
calculated_measurements = populations[:, 0]
measurement_error = 1.0 / num_shot
cost = 0.5 * graph.sum(((calculated_measurements - measurement_batch) / measurement_error) ** 2.0)
cost.name = "cost"
hessian = graph.hessian(cost, [alpha, delta, gamma], name="hessian")


with QCTRL_HANDLE.parallel():
    tmp0 = dict(graph=graph, cost_node_name="cost", output_node_names=["alpha", "delta", "gamma", "hessian"])
    parallel_results = [QCTRL_HANDLE.functions.calculate_stochastic_optimization(**tmp0) for _ in range(5)]
result = min(parallel_results, key=lambda r: r.best_cost)

# Calculate 2-sigma uncertainties (with 95% precision)
hessian = result.best_output["hessian"]["value"]
uncertainties = 2.0 * np.sqrt(np.diag(np.linalg.inv(hessian)))
alpha_uncertainty, delta_uncertainty, gamma_uncertainty = uncertainties

# Print parameter estimates
print(f"alpha:\t   actual =  {actual_alpha / 1e6:.4f} MHz")
print(f"\testimated = ({result.best_output['alpha']['value'] / 1e6:.4f} "
    f"± {alpha_uncertainty / 1e6:.4f}) MHz"
)
print()
print(f"delta:\t   actual =  {actual_delta / 1e6:.4f} MHz")
print(f"\testimated = ({result.best_output['delta']['value'] / 1e6:.4f} "
    f"± {delta_uncertainty / 1e6:.4f}) MHz"
)
print()
print(f"gamma:\t   actual =  {actual_gamma / 1e6:.4f} MHz")
print(f"\testimated = ({result.best_output['gamma']['value'] / 1e6:.4f} "
    f"± {gamma_uncertainty / 1e6:.4f}) MHz"
)
