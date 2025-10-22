import numpy as np
import stim
import beliefmatching

np_rng = np.random.default_rng()


def demo_weired_DETECTOR():
    circ = stim.Circuit(
    '''
    X 0
    X_ERROR(0.1) 0
    M 0
    DETECTOR rec[-1]
    ''')
    x0 = circ.detector_error_model()
    x1 = beliefmatching.detector_error_model_to_check_matrices(x0)
    x1.check_matrix.toarray()
    x2 = circ.compile_detector_sampler().sample(10) #should Raise Error?


sim = stim.FlipSimulator(batch_size=2**10)
sim.do(stim.Circuit("M(0.1) 0 1"))
sim.num_qubits #1
x0 = sim.get_measurement_flips() #(np,bool,(2,1024))
x0.mean(axis=1) #around 0.1


tmp0 = '''
X_ERROR(1) 0 1 3
REPEAT 5 {
    H 0
    C_XYZ 1
}
'''
circ = stim.Circuit(tmp0)
circ[0] #X_ERROR(1) 0 1 3
circ[1] #REPEAT 5 {...}

sim = stim.FlipSimulator(batch_size=1, disable_stabilizer_randomization=True)
sim.do(circ)
sim.peek_pauli_flips()

sim.do(circ[0])
sim.peek_pauli_flips()

sim.do(circ[1])
sim.peek_pauli_flips()


circ = stim.Circuit(
'''
X_ERROR(0.1) 0 1
H 0
CNOT 0 1
M 0 1
OBSERVABLE_INCLUDE(0) rec[-1] rec[-2]
''')
_,x0 = circ.compile_detector_sampler().sample(10, separate_observables=True)

circ = stim.Circuit(
'''
H 0
CX 0 1
CX 0 2
X_ERROR[E123](0.1) 0 1 2
H 3
CZ 3 0 3 1
MX[E4](0.1) 3
DETECTOR[m1] rec[-1]
H 4
CZ 4 1 4 2
MX[E5](0.1) 4
DETECTOR[m2] rec[-1]
X_ERROR[E678](0.1) 0 1 2
R 3 #reset to |0>
H 3
CZ 3 0 3 1
MX[E9](0.1) 3
DETECTOR[m3] rec[-1]
R 4 #reset to |0>
H 4
CZ 4 1 4 2
MX[E10](0.1) 4
DETECTOR[m4] rec[-1]
X_ERROR[E111213](0.1) 0 1 2
# M 0 1 2
# OBSERVABLE_INCLUDE rec[-3] rec[-2] rec[-1]
''')
x0 = circ.detector_error_model(decompose_errors=True)
x1 = beliefmatching.detector_error_model_to_check_matrices(x0)
