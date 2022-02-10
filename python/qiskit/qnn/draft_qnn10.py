# https://qiskit.org/textbook/ch-applications/image-processing-frqi-neqr.html
import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import qiskit
import qiskit.providers.aer

aer_qasm_sim = qiskit.providers.aer.QasmSimulator()
aer_state_sim = qiskit.providers.aer.StatevectorSimulator()
qi_sv = qiskit.utils.QuantumInstance(aer_state_sim)
qi_qasm = qiskit.utils.QuantumInstance(aer_qasm_sim, shots=8092)


def frqi_state_4pixel(with_measure=True):
    theta = qiskit.circuit.ParameterVector('theta', 4)
    qc = qiskit.QuantumCircuit(3)
    qc.h(0)
    qc.h(1)

    # pixel-0
    qc.cry(theta[0], 0, 2)
    qc.cx(0, 1)
    qc.cry(-theta[0], 1, 2)
    qc.cx(0, 1)
    qc.cry(theta[0], 1, 2)

    # pixel-1
    qc.x(1)
    qc.cry(theta[1], 0, 2)
    qc.cx(0, 1)
    qc.cry(-theta[1], 1, 2)
    qc.cx(0, 1)
    qc.cry(theta[1], 1, 2)

    #pixel-3
    qc.x(1)
    qc.x(0)
    qc.cry(theta[2], 0, 2)
    qc.cx(0, 1)
    qc.cry(-theta[2], 1, 2)
    qc.cx(0, 1)
    qc.cry(theta[2], 1, 2)

    #pixel-4
    qc.x(1)
    qc.cry(theta[3], 0, 2)
    qc.cx(0,1)
    qc.cry(-theta[3], 1, 2)
    qc.cx(0,1)
    qc.cry(theta[3], 1, 2)
    if with_measure:
        qc.measure_all()
    return theta, qc

def get_run_result(qc):
    qobj = qiskit.assemble(qiskit.transpile(qc, aer_qasm_sim), shots=4096)
    counts = aer_qasm_sim.run(qobj).result().get_counts(qc)
    return counts

_, qc = frqi_state_4pixel()
count0 = get_run_result(qc.assign_parameters(np.zeros(4))) #all pixels black
# {'010': 1020, '000': 1051, '001': 1009, '011': 1016}
count1 = get_run_result(qc.assign_parameters(np.ones(4)*np.pi/2)) #all pixels white
# {'100': 946, '101': 1056, '111': 1042, '110': 1052}
# qiskit.visualization.plot_histogram(count0)
