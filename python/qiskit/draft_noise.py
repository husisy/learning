# https://qiskit.org/documentation/tutorials/simulators/3_building_noise_models.html
import numpy as np
import qiskit
import qiskit.providers.aer
import qiskit.tools.visualization

hf_verify_kop = lambda kop: np.abs(sum((np.conj(x.T) @ x) for x in kop)-np.eye(kop[0].shape[0])).max()

bit_flip = qiskit.providers.aer.noise.pauli_error([('X', 0.05), ('I', 0.95)])
phase_flip = qiskit.providers.aer.noise.pauli_error([('Z', 0.05), ('I', 0.95)])
kop0 = bit_flip.compose(phase_flip) #on the same qubit
kop1 = phase_flip.tensor(bit_flip) #on different qubit
np_kop0 = qiskit.quantum_info.Kraus(kop0)
kop2 = qiskit.providers.aer.noise.QuantumError(np_kop0)
assert hf_verify_kop(np_kop0.data) < 1e-7
kop0_sop = qiskit.quantum_info.SuperOp(kop0) #Superoperator

# readout error
p0given1 = 0.1
p1given0 = 0.05
qiskit.providers.aer.noise.ReadoutError([[1 - p1given0, p1given0], [p0given1, 1 -p0given1]])

# Add depolarizing error to all single qubit u1, u2, u3 gates
noise_model = qiskit.providers.aer.noise.NoiseModel()
error = qiskit.providers.aer.noise.depolarizing_error(0.05, 1)
noise_model.add_all_qubit_quantum_error(error, ['u1', 'u2', 'u3'])

# Add depolarizing error to all single qubit u1, u2, u3 gates on qubit 0 only
noise_model = qiskit.providers.aer.noise.NoiseModel()
error = qiskit.providers.aer.noise.depolarizing_error(0.05, 1)
noise_model.add_quantum_error(error, ['u1', 'u2', 'u3'], [0])

# Add depolarizing error on qubit 2 forall single qubit u1, u2, u3 gates on qubit 0
noise_model = qiskit.providers.aer.noise.NoiseModel()
error = qiskit.providers.aer.noise.depolarizing_error(0.05, 1)
noise_model.add_nonlocal_quantum_error(error, ['u1', 'u2', 'u3'], [0], [2])


def GHZ_circuit(num_qubit=4):
    circ = qiskit.QuantumCircuit(num_qubit)
    circ.h(0)
    for x in range(num_qubit-1):
        circ.cx(x, x+1)
    circ.measure_all()
    return circ

## common noise model
circ = GHZ_circuit(num_qubit=4)
sim_ideal = qiskit.providers.aer.AerSimulator()
result_ideal = sim_ideal.run(circ).result()
# qiskit.tools.visualization.plot_histogram(result_ideal.get_counts(0))

# noise
circ = GHZ_circuit(num_qubit=4)
p_reset = 0.03
p_meas = 0.1
p_gate1 = 0.05
error_reset = qiskit.providers.aer.noise.pauli_error([('X', p_reset), ('I', 1 - p_reset)])
error_meas = qiskit.providers.aer.noise.pauli_error([('X',p_meas), ('I', 1 - p_meas)])
error_gate1 = qiskit.providers.aer.noise.pauli_error([('X',p_gate1), ('I', 1 - p_gate1)])
error_gate2 = error_gate1.tensor(error_gate1)
noise_bit_flip = qiskit.providers.aer.noise.NoiseModel()
noise_bit_flip.add_all_qubit_quantum_error(error_reset, "reset")
noise_bit_flip.add_all_qubit_quantum_error(error_meas, "measure")
noise_bit_flip.add_all_qubit_quantum_error(error_gate1, ["u1", "u2", "u3"])
noise_bit_flip.add_all_qubit_quantum_error(error_gate2, ["cx"])
sim_noise = qiskit.providers.aer.AerSimulator(noise_model=noise_bit_flip)
circ_tnoise = qiskit.transpile(circ, sim_noise)
result_bit_flip = sim_noise.run(circ_tnoise).result()
counts_bit_flip = result_bit_flip.get_counts(0)
# qiskit.tools.visualization.plot_histogram(counts_bit_flip)


## thermal relaxation noise model
# T1 and T2 values for qubits 0-3
num_qubit = 4
circ = GHZ_circuit(num_qubit)
T1s = np.random.normal(50e3, 10e3, num_qubit) # Sampled from normal distribution mean 50 microsec
T2s = np.random.normal(70e3, 10e3, num_qubit)  # Sampled from normal distribution mean 50 microsec
T2s = np.array([min(T2s[j], 2 * T1s[j]) for j in range(4)]) # Truncate random T2s <= T1s
# Instruction times (in nanoseconds)
time_u1 = 0   # virtual gate
time_u2 = 50  # (single X90 pulse)
time_u3 = 100 # (two X90 pulses)
time_cx = 300
time_reset = 1000  # 1 microsecond
time_measure = 1000 # 1 microsecond
errors_reset = [qiskit.providers.aer.noise.thermal_relaxation_error(x,y,time_reset) for x,y in zip(T1s, T2s)]
errors_measure = [qiskit.providers.aer.noise.thermal_relaxation_error(x,y,time_measure) for x,y in zip(T1s, T2s)]
errors_u1 = [qiskit.providers.aer.noise.thermal_relaxation_error(x,y,time_u1) for x,y in zip(T1s, T2s)]
errors_u2 = [qiskit.providers.aer.noise.thermal_relaxation_error(x,y,time_u2) for x,y in zip(T1s, T2s)]
errors_u3 = [qiskit.providers.aer.noise.thermal_relaxation_error(x,y,time_u3) for x,y in zip(T1s, T2s)]
errors_cx = [[qiskit.providers.aer.noise.thermal_relaxation_error(y0,y1,time_cx).expand(
            qiskit.providers.aer.noise.thermal_relaxation_error(x0,x1,time_cx)) for y0,y1 in zip(T1s, T2s)] for x0,x1 in zip(T1s, T2s)]
noise_thermal = qiskit.providers.aer.noise.NoiseModel()
for j in range(4):
    noise_thermal.add_quantum_error(errors_reset[j], "reset", [j])
    noise_thermal.add_quantum_error(errors_measure[j], "measure", [j])
    noise_thermal.add_quantum_error(errors_u1[j], "u1", [j])
    noise_thermal.add_quantum_error(errors_u2[j], "u2", [j])
    noise_thermal.add_quantum_error(errors_u3[j], "u3", [j])
    for k in range(4):
        noise_thermal.add_quantum_error(errors_cx[j][k], "cx", [j, k])
sim_thermal = qiskit.providers.aer.AerSimulator(noise_model=noise_thermal)
circ_tthermal = qiskit.transpile(circ, sim_thermal)
result_thermal = sim_thermal.run(circ_tthermal).result()
counts_thermal = result_thermal.get_counts(0)
# qiskit.tools.visualization.plot_histogram(counts_thermal)
