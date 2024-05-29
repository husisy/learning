import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import openqaoa
# from openqaoa.problems import MinimumVertexCover

g = nx.circulant_graph(6, [1])
vc = openqaoa.problems.MinimumVertexCover(g, field=1.0, penalty=10)
qubo_problem = vc.qubo
q = openqaoa.QAOA()
q.compile(qubo_problem)
q.optimize()
q.circuit_properties.asdict()
q.result.optimized
q.result.optimized['angles']
q.result.optimized['cost']
q.result.optimized['measurement_outcomes']
# job_id
# eval_number
# q.result.intermediate
q.result.plot_cost()
np.abs(q.result.optimized['cost'])
q.result.plot_probabilities()




g = nx.circulant_graph(6, [1])
vc = openqaoa.problems.MinimumVertexCover(g, field=1.0, penalty=10)
qubo_problem = vc.qubo
q = openqaoa.QAOA()
q.set_circuit_properties(p=3, param_type='standard', init_type='ramp', mixer_hamiltonian='xy')
q.set_backend_properties(init_hadamard=True, n_shots=8000, cvar_alpha=0.85)
q.set_classical_optimizer(method='cobyla', maxiter=50, tol=0.05)
q.compile(qubo_problem)
q.optimize()
# Conditional Value-at-Risk (cvar)


# https://openqaoa.entropicalabs.com/problems/what-is-a-qubo/#qubos-in-openqaoa
terms = [[0], [1], [0,1], [1,2], [0,2]]
weights = [3, 2, 6, 4, 5]
qubo = openqaoa.problems.QUBO(n=3, terms=terms, weights=weights)
qubo.hamiltonian.expression
graph = openqaoa.utilities.graph_from_hamiltonian(qubo.hamiltonian)
nx.draw(graph, with_labels=True, node_color='yellow')


from openqaoa.problems import Knapsack

knapsack_prob = openqaoa.problems.Knapsack.random_instance(n_items=4) #seed=42
knapsack_prob.values
knapsack_prob.weights
knapsack_prob.weight_capacity
knapsack_qubo = knapsack_prob.qubo
knapsack_qubo.hamiltonian.expression
knapsack_qubo.asdict()
