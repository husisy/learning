import os
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import qiskit
import qiskit.providers.aer
import qiskit.providers.ibmq

with open(os.path.expanduser('~/qiskit_token.txt'), 'r') as fid:
    IBMQ_TOKEN = fid.read().strip()

ibmq_provider = qiskit.providers.ibmq.IBMQ.enable_account(IBMQ_TOKEN, group='open', hub='ibm-q', project='main')
ibmq_provider = qiskit.providers.ibmq.IBMQ.enable_account(IBMQ_TOKEN, group='qscitech-quantum', hub='ibm-q-education', project='qc-bc-workshop')
# ibmq_qasm_simulator
# ibmq_armonk
# ibmq_santiago
# ibmq_bogota
# ibmq_lima
# ibmq_belem
# ibmq_quito
# simulator_statevector
# simulator_mps
# simulator_extended_stabilizer
# simulator_stabilizer
# ibmq_manila

runtime_backend = ibmq_provider.backends(input_allowed='runtime')
program = ibmq_provider.runtime.program('circuit-runner')
# maximum execution time, parameter descriptions, etc.


# Number of iterations to run. Each iteration generates a runs a random circuit.
runtime_inputs = {'iterations': 1}
options = {'backend_name': 'ibmq_qasm_simulator'}
job = ibmq_provider.runtime.run(
	program_id='hello-world',
	options=options,
	inputs=runtime_inputs,
)
job.job_id() #c7k2gsplkn32a224of3g
job.status() #JobStatus.DONE: 'job has successfully run'
result = job.result() #(str) 'All done!'


pauli_x = qiskit.quantum_info.operators.Pauli(label='X')
runtime_inputs = {
    # The fraction of top measurement samples to be used for the expectation value (CVaR expectation).
    # i.e. using all samples to construct the expectation value. 1(default)
    'alpha': 1,
	# A list of operators to be evaluated at the final, optimized state. This must be a List[PauliSumOp].
    'aux_operators': None, # array
	# Initial parameters of the ansatz. Can be an array or the string ``'random'`` to choose random initial parameters. The type must be numpy.ndarray or str.
    'initial_point': None, # [array,string]
	# Whether to apply measurement error mitigation in form of a tensored measurement fitter to the measurements. False(default)
    'measurement_error_mitigation': False, # boolean
	# The cost Hamiltonian, consisting of Pauli I and Z operators, whose smallest eigenvalue we're trying to find. The type must be a PauliSumOp.
    'operator': pauli_x,
	# The optimization level to run if the swap strategies are not used. 1(default)
    'optimization_level': 1, # integer
	# The classical optimizer used to update the parameters in each iteration. Per default, SPSA with
    # automatic calibration of the learning rate is used. The type must be a qiskit.algorithms.optimizers.Optimizer.
    'optimizer': None, # object
	# The number of QAOA repetitions, i.e. the QAOA depth typically labeled p. 1(default)
    'reps': 5,
	# The integer number of shots used for each circuit evaluation. 1024(default)
    'shots': 1024, # integer
	# A boolean flag that, if set to True, runs a heuristic algorithm to permute the Paulis
    # in the cost operator to better fit the coupling map and the swap strategy. This is only needed when the
    # optimization problem is sparse and when using swap strategies to transpile. False(default)
    'use_initial_mapping': False, # boolean
	# A boolean on whether or not to use a pulse-efficient transpilation. False(default)
    'use_pulse_efficient': False, # boolean
	# A boolean on whether or not to use swap strategies when transpiling.
    # If this is False then the standard transpiler with the given optimization level will run. True(default)
    'use_swap_strategies': True # boolean
}
options = {'backend_name': 'ibmq_qasm_simulator'}
job = ibmq_provider.runtime.run(
	program_id='qaoa',
	options=options,
	inputs=runtime_inputs
)
job.status()
result = job.result()



##
k = 5
qubits = 3
state = np.exp((-2j*np.pi*k) * np.linspace(0, 1, 8, endpoint=False)) / np.sqrt(8)
circuit = qiskit.QuantumCircuit(qubits, qubits)
circuit.initialize(state)
circuit.append(qiskit.circuit.library.QFT(qubits), range(qubits))
circuit.measure(range(qubits), range(qubits))

# ibmq_qasm_simulator ibmq_armonk(fail,coupling is wrong)
result = ibmq_provider.run_circuits(circuit, backend_name='ibmq_qasm_simulator', optimization_level=3).result()
counts = result.get_counts()

program_inputs = {
    'circuits': circuit,
    'optimization_level': 3,
    'measurement_error_mitigation': True, #default False
}
options = {'backend_name': 'ibmq_qasm_simulator'}
job = ibmq_provider.runtime.run(program_id="circuit-runner", options=options, inputs=program_inputs)
job.job_id()
result = job.result(decoder=qiskit.providers.ibmq.RunnerResult)
noisy = result.get_counts()
mitigated = result.get_quasiprobabilities().nearest_probability_distribution()
# qiskit.visualization.plot_histogram([noisy, mitigated], legend=['noisy', 'mitigated'])
