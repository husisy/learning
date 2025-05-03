import os
import numpy as np
import stim
import beliefmatching
import pymatching
import matplotlib.pyplot as plt

def save_circuit(x:str, file='tbd00', suffix='svg'):
    with open(os.path.join([file,suffix]), 'w') as fid:
        fid.write(str(x))


circ = stim.Circuit()
circ.append("H", [0])
circ.append("CNOT", [0, 1]) #bell state
circ.append("M", [0, 1]) #measure
# |00> + |11>
print(circ) #print(repr(circ))
# save_circuit(circ.diagram('timeline-svg'))
# H 0
# CX 0 1
# M 0 1
x0 = circ.compile_sampler().sample(shots=10)
# (np,bool,(5,2))


circ = stim.Circuit()
circ.append("H", [0])
circ.append("CNOT", [0, 1])
circ.append("M", [0, 1])
circ.append('DETECTOR', [stim.target_rec(-1), stim.target_rec(-2)]) #parity check
x0 = circ.compile_detector_sampler().sample(shots=5) #False when (0,0) and (1,1)


circ = stim.Circuit()
circ.append("H", [0])
circ.append("TICK") #progression of the time, for timeslice-svg
circ.append("CNOT", [0, 1])
circ.append("X_ERROR", [0, 1], 0.2) #independent error
circ.append("M", [0, 1])
circ.append('DETECTOR', [stim.target_rec(-1), stim.target_rec(-2)])
x0 = circ.compile_detector_sampler().sample(shots=1000)
x0[:,0].mean() #one error probability: 0.8*0.2*2
# save_circuit(circ.diagram('timeslice-svg'))


circ = stim.Circuit.generated("repetition_code:memory", rounds=25, distance=9,
        before_round_data_depolarization=0.04, before_measure_flip_probability=0.01)
one_sample = circ.compile_sampler().sample(shots=1)[0]
for k in range(0, len(one_sample), 8):
    print("".join("1" if e else "_" for e in one_sample[k:k+8]))

one_sample = circ.compile_detector_sampler().sample(shots=1)[0]
for k in range(0, len(one_sample), 8):
    print("".join("!" if e else "_" for e in one_sample[k:k+8]))

##
circ = stim.Circuit.generated("repetition_code:memory", rounds=25, distance=9,
        before_round_data_depolarization=0.04, before_measure_flip_probability=0.01)
# round: how many times that stabilizers are measured
# save_circuit(circ.diagram('timeline-svg'))
circ.num_qubits #17
circ.num_ticks #75
circ.num_detectors #208
circ.num_measurements #209
one_sample = circ.compile_sampler().sample(shots=1)[0]
for k in range(0, len(one_sample), 8):
    print("".join("1" if e else "_" for e in one_sample[k:k+8]))
# detected events come in pair (except the boundary)
one_sample = circ.compile_detector_sampler().sample(shots=1)[0]
for k in range(0, len(one_sample), 8):
    print("".join("!" if e else "_" for e in one_sample[k:k+8]))



circ = stim.Circuit.generated("repetition_code:memory", rounds=25, distance=9,
        before_round_data_depolarization=0.04, before_measure_flip_probability=0.01)
dem = circ.detector_error_model()
print(repr(dem))
# save_circuit(dem.diagram("matchgraph-svg"))


def count_logical_errors(circ:stim.Circuit, num_shots:int) -> int:
    sampler = circ.compile_detector_sampler()
    detection, observable_flips = sampler.sample(num_shots, separate_observables=True)
    # detection (np,bool,(num_shots,208))
    # observable_flips (np,bool,(num_shots,1))
    matcher = pymatching.Matching.from_detector_error_model(circ.detector_error_model(decompose_errors=True))
    predictions = matcher.decode_batch(detection) #(np,uint8,(num_shots,1))
    num_errors = (predictions!=observable_flips).sum()
    return num_errors

circ = stim.Circuit.generated("repetition_code:memory", rounds=100, distance=9, before_round_data_depolarization=0.03)
num_shots = 100000
num_logical_errors = count_logical_errors(circ, num_shots)
print("there were", num_logical_errors, "wrong predictions (logical errors) out of", num_shots, "shots")




num_shots = 10000
distance_list = [3,5,7]
physical_error_list = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
logical_error_list = []
for d in distance_list:
    for noise in [0.1, 0.2, 0.3, 0.4, 0.5]:
        circ = stim.Circuit.generated("repetition_code:memory", rounds=d*3, distance=d, before_round_data_depolarization=noise)
        logical_error_list.append(count_logical_errors(circ, num_shots))
logical_error_list = np.array(logical_error_list).reshape((len(distance_list), len(physical_error_list)))

fig,ax = plt.subplots()
for ind0 in range(len(distance_list)):
    ax.plot(physical_error_list, logical_error_list[ind0]/num_shots, label=f'd={distance_list[ind0]}')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
fig.savefig('tbd00.png', dpi=200)


circ = stim.Circuit.generated("surface_code:rotated_memory_z", rounds=9, distance=3,
        after_clifford_depolarization=0.001, after_reset_flip_probability=0.001, before_measure_flip_probability=0.001,
        before_round_data_depolarization=0.001)
save_circuit(circ.without_noise().diagram("timeslice-svg"))
# circ.without_noise().diagram("timeline-3d")
# circ.diagram("detslice-svg")
# circ.without_noise().diagram("detslice-with-ops-svg", tick=range(0, 9))
save_circuit(circ.diagram("matchgraph-3d"), suffix='gltf')



circ = stim.Circuit('''
R 0 1
X_ERROR(0.1) 0 1
M 0 1  # This measurement is always False under noiseless execution.
# Annotate that most recent measurement should be deterministic.
DETECTOR rec[-1] rec[-1]
''')
x0 = circ.compile_sampler().sample(shots=10)
x1 = circ.compile_detector_sampler().sample(shots=10)

circ = stim.Circuit('''R 0
X_ERROR(0.1) 0
M 0
''')
x0 = circ.compile_sampler().sample(shots=1000) #(np,bool,(1000,1))
x0.mean() #around 0.1

circ = stim.Circuit('''R 0 1
X_ERROR(0.1) 0 1
M 0 0 1
DETECTOR rec[-2] rec[-1]
DETECTOR rec[-1] rec[-1]
''')
x0 = circ.compile_sampler().sample(shots=1000) #(np,bool,(1000,3))
assert np.all(x0[:,0]==x0[:,1])
x1 = circ.compile_detector_sampler().sample(shots=1000) #(np,bool,(1000,2))
assert np.all(~x1[:,1]) #all False
x1[:,0].mean() #around 0.1*0.9*2


circ = stim.Circuit('''
H 0
CNOT 0 1
''')
x0 = circ.to_tableau()
x0.to_stabilizers() #XX ZZ

circ = stim.Circuit('''
QUBIT_COORDS(0) 0
QUBIT_COORDS(1) 1
QUBIT_COORDS(2) 2
''')
x0 = circ.to_tableau()
x0.to_stabilizers() #ZII IZI IIZ




# https://quantumcomputing.stackexchange.com/q/33657
circ = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=1,
                after_clifford_depolarization=0.005, before_round_data_depolarization=0.005,
                after_reset_flip_probability=0.005, before_measure_flip_probability=0.005)
model = circ.detector_error_model(decompose_errors=True)
H = beliefmatching.detector_error_model_to_check_matrices(model).check_matrix #(scipy.sparse.csc,uint8,(24,77))
circ.num_detectors #24


circ = stim.Circuit.generated("surface_code:rotated_memory_x", distance=5, rounds=5,
                after_clifford_depolarization=0.005, before_round_data_depolarization=0.005,
                after_reset_flip_probability=0.005, before_measure_flip_probability=0.005)
model = circ.detector_error_model(decompose_errors=True)
H = beliefmatching.detector_error_model_to_check_matrices(model).check_matrix #(scipy.sparse.csc,uint8,(120,1679))



circ = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=1,
                after_clifford_depolarization=0.001, before_round_data_depolarization=0.002,
                after_reset_flip_probability=0.003, before_measure_flip_probability=0.004)
circ.to_file('tbd00.txt')


circ = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=2,
                after_clifford_depolarization=0.001, before_round_data_depolarization=0.002,
                after_reset_flip_probability=0.003, before_measure_flip_probability=0.004)
circ.to_file('tbd01.txt')



circ = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=1,
                after_clifford_depolarization=0, before_round_data_depolarization=0,
                after_reset_flip_probability=0, before_measure_flip_probability=0)
z0 = circ.compile_sampler().sample(100)
tmp0 = [[-9,-6,-3], [-8,-5,-2], [-7,-4,-1]] #logical X
for x in tmp0:
    assert np.all(z0[:,x].sum(axis=1) % 2 == 0)
tmp0 = [[-9,-8], [-8,-7,-5,-4], [-6,-5,-3,-2], [-2,-1]] #stabilizer X
for x in tmp0:
    assert np.all(z0[:,x].sum(axis=1) % 2 == 0)
assert np.all(~circ.compile_detector_sampler().sample(100))


circ = stim.Circuit('''
QUBIT_COORDS(1, 1) 1
QUBIT_COORDS(2, 0) 2
QUBIT_COORDS(3, 1) 3
QUBIT_COORDS(5, 1) 5
QUBIT_COORDS(1, 3) 8
QUBIT_COORDS(2, 2) 9
QUBIT_COORDS(3, 3) 10
QUBIT_COORDS(4, 2) 11
QUBIT_COORDS(5, 3) 12
QUBIT_COORDS(6, 2) 13
QUBIT_COORDS(0, 4) 14
QUBIT_COORDS(1, 5) 15
QUBIT_COORDS(2, 4) 16
QUBIT_COORDS(3, 5) 17
QUBIT_COORDS(4, 4) 18
QUBIT_COORDS(5, 5) 19
QUBIT_COORDS(4, 6) 25
R 1 3 5 8 10 12 15 17 19
R 2 9 11 13 14 16 18 25
TICK
H 2 11 16 25
TICK
CX 2 3 16 17 11 12 15 14 10 9 19 18
TICK
CX 2 1 16 15 11 10 8 14 3 9 12 18
TICK
CX 16 10 11 5 25 19 8 9 17 18 12 13
TICK
CX 16 8 11 3 25 17 1 9 10 18 5 13
TICK
H 2 11 16 25
TICK
MR 2 9 11 13 14 16 18 25
MX 1 3 5 8 10 12 15 17 19
DETECTOR(2, 0, 1) rec[-8] rec[-9] rec[-17]
DETECTOR(2, 4, 1) rec[-2] rec[-3] rec[-5] rec[-6] rec[-12]
DETECTOR(4, 2, 1) rec[-4] rec[-5] rec[-7] rec[-8] rec[-15]
DETECTOR(4, 6, 1) rec[-1] rec[-2] rec[-10]
OBSERVABLE_INCLUDE(0) rec[-3] rec[-6] rec[-9]
''')
z0 = circ.compile_sampler().sample(100)


circ = stim.Circuit.generated("surface_code:rotated_memory_x", distance=3, rounds=1,
                after_clifford_depolarization=0.01, before_round_data_depolarization=0.02,
                after_reset_flip_probability=0.02, before_measure_flip_probability=0.03)
circ.to_file('tbd00.txt')

circ = stim.Circuit.from_file('tbd00.txt')
model = circ.detector_error_model(decompose_errors=True)
H = beliefmatching.detector_error_model_to_check_matrices(model).check_matrix #(scipy.sparse.csc,uint8,(120,1679))
print(H.shape)



# TODO what's the check matrix
distance = 3
rounds = 1
num_shot = int(1e5)
noise = 0.001
circ = stim.Circuit.generated("surface_code:rotated_memory_x", distance=distance, rounds=rounds,
        after_clifford_depolarization=noise, before_round_data_depolarization=noise,
        after_reset_flip_probability=noise, before_measure_flip_probability=noise)
matcher = pymatching.Matching.from_detector_error_model(circ.detector_error_model(decompose_errors=True))

model = circ.detector_error_model(decompose_errors=True)
tmp0 = beliefmatching.detector_error_model_to_check_matrices(model)
H = tmp0.check_matrix #(scipy.sparse.csc,uint8,(120,1679))
weights = tmp0.priors
obs = tmp0.observables_matrix
matcher = pymatching.Matching.from_check_matrix(H, weights=weights, faults_matrix=obs)

event, observable_flips = circ.compile_detector_sampler().sample(num_shot, separate_observables=True)
# event (np,bool,shape=(num_shot,num_detectors)) is the measurement result of each detector
# observable_flips (np,bool,shape=(num_shot,1)) is the observable flips of each detector
predictions = matcher.decode_batch(event)
ret = (predictions!=observable_flips).sum()
