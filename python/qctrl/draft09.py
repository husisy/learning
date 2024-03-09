import numpy as np
import matplotlib.pyplot as plt
import dotenv

import qctrl
import qctrlvisualizer

plt.style.use(qctrlvisualizer.get_qctrl_style())

QCTRL_ORGANIZATION = dotenv.dotenv_values('.env')['QCTRL_ORGANIZATION']
QCTRL_HANDLE = qctrl.Qctrl(organization=QCTRL_ORGANIZATION)
QCTRL_HANDLE.request_machines(1)


# https://docs.q-ctrl.com/boulder-opal/user-guides/how-to-calculate-and-optimize-with-graphs
graph = QCTRL_HANDLE.create_graph()
# no need to assign a name to it as we don't want to extract its value
node0 = graph.pauli_kronecker_product([("Z", 0), ("Z", 1)], 2)
node1 = node0 + np.eye(4)
node1.name = "matrix"
node2 = graph.trace(node1, name="trace")
result = QCTRL_HANDLE.functions.calculate_graph(graph=graph, output_node_names=["matrix", "trace"])
result.output.keys()
result.output['matrix']['value']
result.output['trace']['value']


graph = QCTRL_HANDLE.create_graph()
tmp0 = graph.optimization_variable(2, lower_bound=-10, upper_bound=10)
x = tmp0[0]
x.name = "x"
y = tmp0[1]
y.name = "y"
loss = (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2
loss.name = "loss"
result = QCTRL_HANDLE.functions.calculate_optimization(graph=graph,
        cost_node_name="loss", output_node_names=["x", "y"], optimization_count=4)
result.output['y']['value']


# https://docs.q-ctrl.com/boulder-opal/user-guides/how-to-represent-quantum-systems-using-graphs
## optimal control
omega_0 = 2 * np.pi * 0.5e6  #Hz
segment_count = 50
duration = 10e-6  #second
alpha_max = 2 * np.pi * 0.25e6  #Hz
beta = 2 * np.pi * 20e3  #dephasing noise amplitude (Hz)

graph = QCTRL_HANDLE.create_graph()
alpha = graph.utils.real_optimizable_pwc_signal(segment_count=segment_count,
        duration=duration, minimum=-alpha_max, maximum=alpha_max, name=r"$\alpha$")
hamiltonian = 0.5 * omega_0 * graph.pauli_matrix("Z") + alpha * graph.pauli_matrix("X")
dephasing = beta * graph.pauli_matrix("Z")
tmp0 = np.cos(omega_0*duration/2) * graph.pauli_matrix("I") - 1j * np.sin(omega_0*duration/2)*graph.pauli_matrix("Z")
target = graph.target(operator=tmp0)
infidelity = graph.infidelity_pwc(hamiltonian=hamiltonian, noise_operators=[dephasing], target=target, name="infidelity")
result = QCTRL_HANDLE.functions.calculate_optimization(graph=graph, cost_node_name="infidelity", output_node_names=["$\\alpha$"])
result.cost
# qctrlvisualizer.plot_controls(controls=result.output)
